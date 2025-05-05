import imageio
# import napari for data visualisation
import napari
import os
os.environ['OMP_NUM_THREADS'] = '1'

# import the segmentation functionality from elf
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
from elf.segmentation.utils import seg_to_edges

# import the open_file function from elf, which supports opening files
# in hdf5, zarr, n5 or knossos file format
from elf.io import open_file

import tifffile

from elf.segmentation.features import compute_rag
from elf.segmentation import stacked_watershed
from elf.segmentation.watershed import distance_transform_watershed
from elf.segmentation.multicut import (
    multicut_kernighan_lin,
    transform_probabilities_to_costs,
)

import nifty
import nifty.graph.rag as nrag
import elf.segmentation.watershed as ws

import numpy as np
import unittest
from elf.segmentation.watershed import distance_transform_watershed
import skimage
import pandas as pd
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
from pathlib import Path

def run_multicut_callum(prediction):
    """Performs multicut segmentation on a border prediction"""
    ws, _ = distance_transform_watershed(
        prediction,
        threshold=0.7,
        sigma_seeds=.0,
        min_size=30,
        #max_size=200 this parameter is not available to modify in the function
    )

    rag = compute_rag(ws)

    features = nrag.accumulateEdgeMeanAndLength(rag, prediction, numberOfThreads=1)

    # mean edge probability
    probs = features[:, 0]

    edge_sizes = features[:, 1]
    #testing these parameters
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=0.7)

    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)

    graph.insertEdges(rag.uvIds())

    # Solve multicut
    node_labels = multicut_kernighan_lin(graph, costs)
    #check nifty function and uncomment it to test
    final_seg_callum = nifty.tools.take(node_labels, ws)
    final_seg_params_filtered = filter_objects(final_seg_callum, min_size=20, max_size=250)

    return final_seg_params_filtered

def filter_objects(labels,min_size=10, max_size=250):
    df = skimage.measure.regionprops_table(labels, properties=("label","axis_major_length"))
    df = pd.DataFrame.from_dict(df)
    remove_labels = df[
        (df["axis_major_length"] < min_size) | 
        (df["axis_major_length"] > max_size)
    ]["label"].values
    
    mask = np.isin(labels, remove_labels)
    labels[mask] = 0
    
    # Reasignar etiquetas secuenciales
    labels, _, _ = relabel_sequential(labels)
    return labels
    #min_objects = df[(df["axis_major_length"] < min_size)]["label"].values
    #max_objects = df[(df["axis_major_length"] > max_size)]["label"].values
    #remove_labels = np.concatenate((min_objects, max_objects), axis=0)
    #for rmv in remove_labels:
    #    labels = np.where(labels == rmv, 0, labels)
    #labels = make_sequential(labels)
    #return labels

def make_sequential(array):
    unique = np.unique(array)
    array = np.searchsorted(unique, array)
    return array


def run_multicut_test(prediction,configs_ws,beta_values, weighting_exponent=2.0):
    """Performs multicut segmentation on a border prediction"""
    #shape = (22, 1024, 1024)
    #inp = np.random.rand(*shape).astype("float32")
    #initialize the list for results
    results = []
    # Test for different options


    for config in configs_ws:
        ws, _ = distance_transform_watershed(prediction, **config)

        rag = compute_rag(ws)

        features = nrag.accumulateEdgeMeanAndLength(rag, prediction, numberOfThreads=1)

        # Probabilidades de borde
        probs = features[:, 0]
        edge_sizes = features[:, 1]
        edge_sizes_norm = edge_sizes / (edge_sizes.max() + 1e-6)
        
        for beta in beta_values:
            costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes_norm, beta=beta, weighting_exponent=2.5)

            graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
            graph.insertEdges(rag.uvIds())

            # Resolver multicut
            node_labels = multicut_kernighan_lin(graph, costs)

            final_seg_params = nifty.tools.take(node_labels, ws)
            #filter small and big objects:
            final_seg_params_filtered = filter_objects(final_seg_params, min_size=20, max_size=350)
            #print('filter objects has been finished')

            #store results of segmentation:
            
            results.append({
                "config": config,
                "beta": beta,
                "segmentation": final_seg_params_filtered
            })

    return results  # Esto devuelve solo el último resultado. Si quieres todos, usa una lista para almacenarlos.


def plot_segmentation_screenshots(final_seg_params, pmaps_image, filenam_without_ext, z_plane=30, save_path="segmentation_grid.png", dpi=500):
    """
    Display segmentation results with Napari and save organized captures in rows of 3.
    
    Args:
        final_seg_params (list): List of dicts containing segmentations and parameters.
        pmaps_image (ndarray): Input image (e.g., edge probabilities).
        filenam_without_ext (str): Base file name for generating titles.
        z_plane (int): Z-plane to be shown in the image.
        save_path (str): Path where the figure will be saved.
        dpi (int): Resolution of the saved figure.
    """

    viewer = napari.Viewer(show=True)
    viewer.add_image(pmaps_image, name='boundaries')
    viewer.dims.set_point(0, z_plane)

    screenshots = []
    titles = []
    layers = []

    for result in final_seg_params:
        beta = result['beta']
        sigma_seeds = result['config']['sigma_seeds']
        threshold = result['config']['threshold']
        min_size = result['config']['min_size']
        
        config_name = f"{filenam_without_ext}_beta_{beta}_sigma_{sigma_seeds}_threshold_{threshold}_minsize_{min_size}"
        
        layer = viewer.add_labels(result['segmentation'], name=config_name)
        viewer.dims.set_point(0, z_plane)
        
        screenshot = viewer.screenshot(canvas_only=True)
        screenshots.append(screenshot)
        titles.append(config_name)
        layers.append(layer)
        
        layer.visible = False

    # Mostrar en filas de 3
    n = len(screenshots)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(screenshots[i])
        axes[i].set_title(f"{titles[i]} - z={z_plane}", fontsize=8)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

    print(f"Figura guardada en: {save_path}")
