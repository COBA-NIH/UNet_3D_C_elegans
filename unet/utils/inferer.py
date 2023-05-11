import monai
import skimage
from unet.networks.unet3d import UNet3D
import unet.utils.data_utils as utils
import unet.utils.metrics as metrics
import torch
import numpy as np
import pandas as pd
import pathlib
import os
import nifty
import nifty.graph.rag as nrag
from elf.segmentation.features import compute_rag
from elf.segmentation import stacked_watershed
from elf.segmentation.watershed import distance_transform_watershed
from elf.segmentation.multicut import (
    multicut_kernighan_lin,
    transform_probabilities_to_costs,
)
from neptune.new.types import File
import nd2
import numpy as np


class Inferer:
    def __init__(
        self,
        model,
        patch_size,
        batch_size=4,
        overlap=0.75,
        patch_mode="gaussian",
        min_size=15,
        max_size=250,
        threshold=0.5,
        neptune_run=None,
    ):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.patch_mode = patch_mode
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold
        self.neptune_run = neptune_run

        if torch.cuda.is_available():
            # Find fastest conv
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def predict_from_image(self, input_image):
        with torch.no_grad():
            prediction = monai.inferers.sliding_window_inference(
                input_image,
                roi_size=self.patch_size,
                sw_batch_size=self.batch_size,
                predictor=self.model,
                progress=True,
                overlap=self.overlap,
                mode=self.patch_mode,
            )
        return prediction

    def predict_from_csv(self, inference_data_csv, from_nd2=False):
        """Iterate through the first column of the data inference csv"""
        # Create new columns to store prediction and segmentation paths
        inference_data_csv["prediction"] = np.nan
        inference_data_csv["segmentation"] = np.nan
        # Create an output folder of just the predictions and segmentations
        os.makedirs("output", exist_ok=True)
        for i, input_image_path in enumerate(inference_data_csv.iloc[:, 0]):
            if from_nd2:
                input_image = nd2.imread(input_image_path).astype(np.float32)
                # nd2 files have dimension order ZCXY, change to expected CZXY
                input_image = np.swapaxes(input_image, 1, 0)
            else:
                input_image = skimage.io.imread(input_image_path).astype(np.float32)
            input_image = torch.from_numpy(input_image)
            input_image = input_image.unsqueeze(0)  # Add batch dimension
            input_image = input_image.to(self.device)

            print(f"Predicting: {input_image_path}")
            prediction = (
                self.predict_from_image(input_image).cpu().sigmoid().numpy()[0, 0, ...]
            )  # Remove batch and channel dimensions

            print(f"Multicutting: {input_image_path}")
            segmentation = self.run_multicut(prediction)
            if self.neptune_run:
                self.neptune_run["segmentation"].upload(File.as_image(segmentation))

            save_path = pathlib.Path(input_image_path)
            prediction_fn = os.path.join(
                "./output/", f"{save_path.stem}_prediction.tiff"
            )
            segmentation_fn = os.path.join(
                "./output/", f"{save_path.stem}_segmentation.tiff"
            )
            # Add filenames to output csv
            inference_data_csv.loc[i, "prediction"] = prediction_fn
            inference_data_csv.loc[i, "segmentation"] = segmentation_fn
            # Save output
            skimage.io.imsave(
                prediction_fn, prediction.astype(np.float16), check_contrast=False, compression=("zlib", 1)
            )
            # Filter out large and small objects
            segmentation = utils.filter_objects(
                segmentation,
                min_size=self.min_size,
                max_size=self.max_size 
                )
            
            skimage.io.imsave(
                segmentation_fn,
                segmentation.astype(np.uint16),
                check_contrast=False,
                compression=("zlib", 1),
            )

        inference_data_csv = self.calculate_prediction_performance(inference_data_csv)

        inference_data_csv.to_csv("./output/inference_data.csv", index=False)

    def calculate_prediction_performance(self, df):
        # Select rows that have a corresponding mask
        df_masks = df[(~(df["masks"].isna()))].reset_index(drop=True)
        # Calculate IoU threshold
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # Dataframe to hold statistic results that will be concatenatd with df
        stats = pd.DataFrame()
        for img_ind in range(len(df_masks)):
            print(f"Calculating stats for: {df_masks.loc[img_ind, 'masks']}")
            gt_mask = skimage.io.imread(df_masks.loc[img_ind, "masks"])
            pred = skimage.io.imread(df_masks.loc[img_ind, "segmentation"])
            # Filter segmentation
            pred = utils.filter_objects(pred)
            # Dataframe to hold IoU threshold stats
            th_stats = pd.DataFrame()
            for th in iou_thresholds:
                # evaluate_segmentation returns a dictionary with information on F1, TP FN etc.
                iou = metrics.evaluate_segmentation(
                    gt_mask, pred, threshold=th
                )
                # Drop the threshold key from the dictionary
                iou.pop("threshold", None)
                # Convert the dictionary to a DataFrame, rename columns with relevant information
                iou = pd.DataFrame(
                    {k + f"_threshold_{th}": v for k, v in iou.items()},
                    index=[0],
                )
                if self.neptune_run is not None:
                    self.neptune_run[f"evaluation/f1_th_{th}"] = iou[
                        f"f1_threshold_{th}"
                    ].values
                # Concatenate the columns from different stats so they're in the same row
                th_stats = pd.concat([th_stats, iou], axis=1)

            # Calculate adapted_rand_error for each image
            rand_error = metrics.calculate_rand_error(pred, gt_mask)[0]
            th_stats["rand_error"] = rand_error
            if self.neptune_run is not None:
                self.neptune_run[f"evaluation/rand_error"] = rand_error
            # Append row, th_stats, containing iou and rand error
            stats = pd.concat([stats, th_stats], axis=0)
        # Concatenate the stats columns onto the output df
        df_masks = pd.concat([df_masks, stats], axis=1)

        # Append pred + mask rows with non-GT containing rows
        df = pd.concat([df_masks, df[((df["masks"].isna()))].reset_index(drop=True)])

        if self.neptune_run is not None:
            self.neptune_run["evaluation/predictions"].upload(File.as_html(df))


        return df

    def run_multicut(self, prediction):
        """Performs multicut segmentation on a border prediction"""
        ws, _ = distance_transform_watershed(
            prediction,
            threshold=self.threshold,
            sigma_seeds=2.0,
            min_size=self.min_size
        )

        rag = compute_rag(ws)

        features = nrag.accumulateEdgeMeanAndLength(rag, prediction, numberOfThreads=1)

        # mean edge probability
        probs = features[:, 0]

        edge_sizes = features[:, 1]

        costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=0.5)

        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)

        graph.insertEdges(rag.uvIds())

        # Solve multicut
        node_labels = multicut_kernighan_lin(graph, costs)

        final_seg = nifty.tools.take(node_labels, ws)

        return final_seg
