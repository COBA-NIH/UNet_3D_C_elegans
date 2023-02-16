import monai
import skimage
from unet.networks.unet3d import UNet3D
import unet.utils.data_utils as utils
import torch
import numpy as np
import pathlib
import os
import nifty
import nifty.graph.rag as nrag
from elf.segmentation.features import compute_rag
from elf.segmentation.watershed import distance_transform_watershed
from elf.segmentation.multicut import (
    multicut_kernighan_lin,
    transform_probabilities_to_costs,
)

class Inferer:
    def __init__(self, model, patch_size, batch_size=4, overlap=0.75, patch_mode="gaussian"):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.patch_mode = patch_mode
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
                mode=self.patch_mode
            )
        return prediction
    
    def predict_from_csv(self, inference_data_csv):
        """Iterate through the first column of the data inference csv"""
        # Create new columns to store prediction and segmentation paths
        inference_data_csv["prediction"] = np.nan
        inference_data_csv["segmentation"] = np.nan
        # Create an output folder of just the predictions and segmentations
        os.makedirs("output", exist_ok=True)
        for i, input_image_path in enumerate(inference_data_csv.iloc[:,0]):
            input_image = skimage.io.imread(input_image_path).astype(np.float32)
            input_image = torch.from_numpy(input_image)
            input_image = input_image.unsqueeze(0) # Add batch dimension
            input_image = input_image.to(self.device)

            print(f"Predicting: {input_image_path}")
            prediction = self.predict_from_image(input_image).cpu().sigmoid().numpy()[0,0,...] # Remove batch and channel dimensions

            print(f"Multicutting: {input_image_path}")
            segmentation = self.run_multicut(prediction)

            save_path = pathlib.Path(input_image_path)
            prediction_fn = os.path.join("./output/", f"{save_path.stem}_prediction{save_path.suffix}")
            segmentation_fn = os.path.join("./output/", f"{save_path.stem}_labelled{save_path.suffix}")
            inference_data_csv.loc[i, "prediction"] = prediction_fn
            inference_data_csv.loc[i, "segmentation"] = segmentation_fn
            skimage.io.imsave(prediction_fn, prediction, check_contrast=False, compression=("zlib", 1))
            skimage.io.imsave(segmentation_fn, segmentation, check_contrast=False, compression=("zlib", 1))
        
        inference_data_csv.to_csv("./output/inference_data.csv", index=False)

    def run_multicut(self, prediction, planewise=False):
        """Performs multicut segmentation on a border prediction"""
        if planewise:
            ws_kwargs = dict(threshold=0.5, sigma_seeds=2.0,
                    #  sigma_weights=sigma_weights,
                        min_size=15,
                    #  pixel_pitch=pixel_pitch,
                    #  apply_nonmax_suppression=apply_nonmax_suppression,
                    #  mask=mask
                        )
            ws, _ = stacked_watershed(
                prediction,
                ws_function=distance_transform_watershed,
                **ws_kwargs)
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