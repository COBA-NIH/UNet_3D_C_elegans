import monai
import skimage
from unet.networks.unet3d import UNet3D
import unet.utils.data_utils as utils
import unet.utils.segmentation_metrics as metrics
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


class Inferer:
    def __init__(
        self,
        model,
        patch_size,
        batch_size=4,
        overlap=0.75,
        patch_mode="gaussian",
        neptune_run=None,
    ):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.patch_mode = patch_mode
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

    def predict_from_csv(self, inference_data_csv):
        """Iterate through the first column of the data inference csv"""
        # Create new columns to store prediction and segmentation paths
        inference_data_csv["prediction"] = np.nan
        inference_data_csv["segmentation_3d_ws"] = np.nan
        inference_data_csv["segmentation_planewise_ws"] = np.nan
        # Create an output folder of just the predictions and segmentations
        os.makedirs("output", exist_ok=True)
        for i, input_image_path in enumerate(inference_data_csv.iloc[:, 0]):
            input_image = skimage.io.imread(input_image_path).astype(np.float32)
            input_image = torch.from_numpy(input_image)
            input_image = input_image.unsqueeze(0)  # Add batch dimension
            input_image = input_image.to(self.device)

            print(f"Predicting: {input_image_path}")
            prediction = (
                self.predict_from_image(input_image).cpu().sigmoid().numpy()[0, 0, ...]
            )  # Remove batch and channel dimensions

            print(f"Multicutting: {input_image_path}")
            segmentation_3d = self.run_multicut(prediction, planewise=False)
            segmentation_planewise = self.run_multicut(prediction, planewise=True)

            save_path = pathlib.Path(input_image_path)
            prediction_fn = os.path.join(
                "./output/", f"{save_path.stem}_prediction{save_path.suffix}"
            )
            segmentation_3d_fn = os.path.join(
                "./output/", f"{save_path.stem}_labelled_3d_ws{save_path.suffix}"
            )
            segmentation_planewise_fn = os.path.join(
                "./output/", f"{save_path.stem}_labelled_planewise_ws{save_path.suffix}"
            )
            # Add filenames to output csv
            inference_data_csv.loc[i, "prediction"] = prediction_fn
            inference_data_csv.loc[i, "segmentation_3d_ws"] = segmentation_3d_fn
            inference_data_csv.loc[
                i, "segmentation_planewise_ws"
            ] = segmentation_planewise_fn
            # Save output
            skimage.io.imsave(
                prediction_fn, prediction, check_contrast=False, compression=("zlib", 1)
            )
            skimage.io.imsave(
                segmentation_3d_fn,
                segmentation_3d,
                check_contrast=False,
                compression=("zlib", 1),
            )
            skimage.io.imsave(
                segmentation_planewise_fn,
                segmentation_planewise,
                check_contrast=False,
                compression=("zlib", 1),
            )

        inference_data_csv = self.calculate_prediction_performance(inference_data_csv)

        inference_data_csv.to_csv("./output/inference_data.csv", index=False)

    def calculate_prediction_performance(self, df):
        # Select rows that have a corresponding mask
        df = df[(~(df["masks"].isna()))].reset_index(drop=True)
        # Calculate IoU threshold
        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # Dataframe to hold statistic results that will be concatenatd with df
        stats = pd.DataFrame()
        for img_ind in range(len(df)):
            gt_mask = skimage.io.imread(df.loc[img_ind, "masks"])
            pred_3d_mask = skimage.io.imread(df.loc[img_ind, "segmentation_3d_ws"])
            pred_planewise_mask = skimage.io.imread(
                df.loc[img_ind, "segmentation_planewise_ws"]
            )
            # Filter segmentation
            pred_3d_mask = utils.filter_objects(pred_3d_mask)
            pred_planewise_mask = utils.filter_objects(pred_planewise_mask)
            # Dataframe to hold IoU threshold stats
            th_stats = pd.DataFrame()
            for th in iou_thresholds:
                # evaluate_segmentation returns a dictionary with information on F1, TP FN etc.
                iou_3d_ws = metrics.evaluate_segmentation(
                    gt_mask, pred_3d_mask, threshold=th
                )
                iou_planewise_ws = metrics.evaluate_segmentation(
                    gt_mask, pred_planewise_mask, threshold=th
                )
                # Drop the threshold key from the dictionary
                iou_3d_ws.pop("threshold", None)
                iou_planewise_ws.pop("threshold", None)
                # Convert the dictionary to a DataFrame, rename columns with relevant information
                iou_3d_ws = pd.DataFrame(
                    {k + f"_threshold_{th}_3d_ws": v for k, v in iou_3d_ws.items()},
                    index=[0],
                )
                iou_planewise_ws = pd.DataFrame(
                    {
                        k + f"_threshold_{th}_planewise_ws": v
                        for k, v in iou_planewise_ws.items()
                    },
                    index=[0],
                )
                if self.neptune_run is not None:
                    self.neptune_run[f"evaluation/f1_th_{th}_3d_ws"] = iou_3d_ws[
                        f"f1_threshold_{th}_3d_ws"
                    ].values
                    self.neptune_run[f"evaluation/f1_th_{th}_planewise_ws"] = iou_planewise_ws[
                        f"f1_threshold_{th}_planewise_ws"
                    ].values
                # Concatenate the columns from different stats so they're in the same row
                th_stats = pd.concat([th_stats, iou_3d_ws, iou_planewise_ws], axis=1)
            # Append row
            stats = pd.concat([stats, th_stats], axis=0)
        # Calculate adapted_rand_error
        # Concatenate the stats columns onto the output df
        df = pd.concat([df, stats], axis=1)

        if self.neptune_run is not None:
            self.neptune_run["evaluation/predictions"].upload(File.as_html(df))

        return df

    def run_multicut(self, prediction, planewise=False):
        """Performs multicut segmentation on a border prediction"""
        if planewise:
            ws_kwargs = dict(
                threshold=0.5,
                sigma_seeds=2.0,
                #  sigma_weights=sigma_weights,
                min_size=15,
                #  pixel_pitch=pixel_pitch,
                #  apply_nonmax_suppression=apply_nonmax_suppression,
                #  mask=mask
            )
            ws, _ = stacked_watershed(
                prediction, ws_function=distance_transform_watershed, **ws_kwargs
            )
        else:
            ws, _ = distance_transform_watershed(
                prediction,
                threshold=0.5,
                sigma_seeds=2.0,
                min_size=15,
                # sigma_weights=2.0
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