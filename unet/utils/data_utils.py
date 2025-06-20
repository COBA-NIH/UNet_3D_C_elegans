import numpy as np
import scipy
import skimage
import torch
import pathlib
import torch.nn.functional as F
import os
import monai
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops_table, regionprops, label
from scipy.ndimage import find_objects
from scipy.ndimage import binary_dilation, generate_binary_structure
# import mahotas

def generate_patches(image, patch_shape, stride_shape, unfold_dims=[0, 1, 2]):
    """Uses PyTorch unfolde to generate non-overlapping patches
    for a given input in 3D.
    
    Patch shape and stride shape are in order (D, W, H). 

    Output tensor with order: (patch, D, W, H).

    unfold_dims: the dimensions on which to unfold ()
    """
    # For now, patch and stride shape are the same.
    # stride_shape = patch_shape

    if not torch.is_tensor(image):
        raise TypeError("Input is not a Tensor.")

    d0, d1, d2 = unfold_dims

    # Check that the image dimensions divide cleanly into the patch shape
    # If not, pad the image. 
    if any([
        image.shape[d0] % patch_shape[0],
        image.shape[d1] % patch_shape[1],
        image.shape[d2] % patch_shape[2]]):
        print("Patches do not divide by the image shape. Padding image.")
        image = F.pad(
            image,
            (image.size(d2)%patch_shape[2] // 2, image.size(d2)%patch_shape[2] // 2,
            image.size(d1)%patch_shape[1] // 2, image.size(d1)%patch_shape[1] // 2,
            image.size(d0)%patch_shape[0] // 2, image.size(d0)%patch_shape[0] // 2)
            )
    # Add an extra dimension that will hold the patches
    # Add dimension to first axis in the dimensions to unfold
    image = torch.unsqueeze(image, axis=unfold_dims[0])
    # Unfold the 1st dimension with size patch_shape[0] with stride_shape[0]
    # Unfold slides along in the provided dimension providing the desired patches
    patches = image.unfold(
        d0+1, patch_shape[0], stride_shape[0]
        ).unfold(
            d1+1, patch_shape[1], stride_shape[1]
            ).unfold(
                d2+1, patch_shape[2], stride_shape[2]
                )
    # Perhaps an unreliable way to infer that there's a channel_dimension that should be respected
    if unfold_dims != [0, 1, 2]:
        # There's a channel dimension 
        patches = patches.contiguous().view(image.size(0), -1, patch_shape[0], patch_shape[1], patch_shape[2]) 
    else:
        # No channel dim, so infer 0th shape
        patches = patches.contiguous().view(-1, patch_shape[0], patch_shape[1], patch_shape[2]) 
    return patches

def save_patches(patches, save_filename, save_dir, patch_dim=0):
    """
    Takes Tensors with shape (patch, D, W, H) and saves them as 
    .tiff files in save_dir"""
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    patches = patches.numpy()

    all_paths = []
    
    for i in range(patches.shape[patch_dim]):
        save_path = pathlib.Path(save_filename)
        out_filename = save_path.with_name(f"{save_path.stem}_patch{i+1}{save_path.suffix}")
        out_path = os.path.join(save_dir, out_filename)
        if patch_dim == 0:
            skimage.io.imsave(out_path, patches[i,...], compression=('zlib', 1))
        else:
            # There's a channel dim, so patch_dim is 1
            skimage.io.imsave(out_path, patches[:,i,...], compression=('zlib', 1))
        all_paths.append(out_path)
    return all_paths


def auto_class_weights(mask, one_hot=True):
    """For a given mask, determine the weight to give each class
    dependent upon the frequency of a given class.
    
    if one_hot == True, it is assumed that the input mask is 
    shape (class, spatial).

    The most abundant class will be asigned a weight of 1.0
    """
    wc = {}
    if one_hot:
        assert len(mask.shape) == 4, "Are you sure this is a 3D one-hot encoded mask?"
        # Assumes no pixels are shared between classes
        mask = np.argmax(mask, axis=0)
    # Grab classes and their corresponding counts
    unique, counts = np.unique(mask, return_counts=True)
    # For each count, divide it by the total number of pixels to get the frequency
    counts = counts / np.product(mask.shape)
    max_count = max(counts)
    for val, count in zip(unique, counts):
        # Find the weight relative to the max count
        wc[val] = max_count / count
    return wc

def calculate_weight_map(gt_array, centroid_class_index=2, edge_class_index=1, labels=None, wc=None, w0=10, sigma=5, background_class=0):
    """
    gt_array: GT pixel classes (eg. background: 0, centroids: 1, cell_edges: 2) with shape (classes, spatial)
    one_hot: True if gt_array is one-hot encoded
    labels: Label map where each object has a unique number. This will allow preservation of the borders defined in GT. 
    edge_class_index: the index on which to find separation borders. If None, will perform on all
    """

    # if not one_hot:

    # if edge_class_index is None:
    #     edge_class_index = slice(0, gt_array.shape[0])

    if wc is None:
        wc = auto_class_weights(gt_array)
    
    if labels is None:
        labels = skimage.measure.label(gt_array[centroid_class_index])

    # Get total number of objects
    objects = np.unique(labels)
    objects = [obj for obj in objects if obj != background_class]

    w = np.zeros_like(gt_array)
    
    # There exists multiple objects
    if len(objects) > 1:
        distances = np.zeros((len(objects), ) + labels.shape)
        for i, region_id in enumerate(objects):
            # Find the distance transforms for all objects except the iterable
            distances[i,...] = scipy.ndimage.distance_transform_edt(labels != region_id)
        """
        Now, np.sort performs the magic of this function.

        Above, you have defined the inverse distance transforms for each label independently. 
        That is, each object will have a distance transform value of 0 and the non-object pixels
        that are closest to the object of interest have increasing values

        Across all of these object distance maps (axis 0), np.sort will find the lowest to 
        highest distance values. Now, since each object was independently set to False for 
        the distance transforms, all elements inside objects are 0 in the 0th (that is, (0, Z, X, Y)). 
        So, for each object you can find distance to the nearest other object. 

        The same is true for the 1st dimension (1, Z, X, Y), which is the distance to the border
        of the second nearest object.

        Analogy: You're at the bottom of a valley and you want to find the closest valley.
        You climb up a small hill to get a better view and indeed, you spot another valley
        on the other side of the small hill (0th). But now you want to find not a valley, but another
        point that has the same height at the point you are at, so you ascend further up another 
        small hill and now you have a better perspective and you look down around you
        and see another point that was a similar height to your first ascent (1st). (ie. you're 
        incrementally looking for local minima).
        """
        distances = np.sort(distances, axis = 0)
        d1, d2 = distances[0,...], distances[1,...]

        # Determine the separation border, as defined in https://arxiv.org/pdf/1505.04597.pdf
        # We multiply by the centroid == background array so that we filter out any distances 
        # that may bleed into the centroid - we want the distance information to only be 
        # in the edge and background classes (I think). 
#        for obj in objects:
 #            size = np.sum(labels == obj)  # Calculate the size of the cell
  #            scaling_factor = calculate_scaling_factor(size)  # Function to determine scaling based on size
   #         w[edge_class_index] += scaling_factor * w0 * np.exp(-1 * ((d1 + d2) ** 2) / (2 * sigma ** 2)) * (gt_array[edge_class_index] > 0.5).astype(np.uint8)
        w[edge_class_index] = w0 * np.exp(-1 * ((d1 + d2) ** 2) / (2 * sigma ** 2)) * (gt_array[edge_class_index] > 0.5).astype(np.uint8)
        #  * (gt_array[centroid_class_index] == background_class).astype(np.uint8)
    # Array to hold class weights
    wc_x = np.zeros_like(gt_array)
    
    # Compute class weights for each pixel class (background, etc.)
    for pixel_class, weight in wc.items():
        wc_x[pixel_class] = np.where(gt_array[pixel_class] == 1, weight, 0)
    
    # # Add them to the weight map
    wc_x = w + wc_x
    
    return wc_x

def calculate_scaling_factor(cell_size, base_factor=1.0, exponent=0.5):
    # Example scaling factor: Increase weight by size^exponent
    return base_factor * (cell_size ** exponent)

def make_sequential(array):
    unique = np.unique(array)
    array = np.searchsorted(unique, array)
    return array

def calculate_binary_weight_map(labels, w0=10, sigma=5):
    # Get total number of objects
    labels = make_sequential(labels)
    objects = np.unique(labels)
    # print(len(objects))
    objects = [obj for obj in objects if obj != 0]
    if len(objects) > 150:
        return labels
    
    # There exists multiple objects
    if len(objects) > 1:
        distances = np.zeros((len(objects), ) + labels.shape)
        for i, region_id in enumerate(objects):
            # Find the distance transforms for all objects except the iterable
            distances[i,...] = scipy.ndimage.distance_transform_edt(labels.copy() != region_id)
        """
        Now, np.sort performs the magic of this function.

        Above, you have defined the inverse distance transforms for each label independently. 
        That is, each object will have a distance transform value of 0 and the non-object pixels
        that are closest to the object of interest have increasing values

        Across all of these object distance maps (axis 0), np.sort will find the lowest to 
        highest distance values. Now, since each object was independently set to False for 
        the distance transforms, all elements inside objects are 0 in the 0th (that is, (0, Z, X, Y)). 
        So, for each object you can find distance to the nearest other object. 

        The same is true for the 1st dimension (1, Z, X, Y), which is the distance to the border
        of the second nearest object.

        Analogy: You're at the bottom of a valley and you want to find the closest valley.
        You climb up a small hill to get a better view and indeed, you spot another valley
        on the other side of the small hill (0th). But now you want to find not a valley, but another
        point that has the same height at the point you are at, so you ascend further up another 
        small hill and now you have a better perspective and you look down around you
        and see another point that was a similar height to your first ascent (1st). (ie. you're 
        incrementally looking for local minima).
        """
        distances = np.sort(distances, axis = 0)
        d1, d2 = distances[0,...], distances[1,...]

        # Determine the separation border, as defined in https://arxiv.org/pdf/1505.04597.pdf
        w = w0 * np.exp(-1 * ((d1 + d2) ** 2) / (2 * sigma ** 2))
    else:
        w = np.zeros_like(labels)
    
    return w


def load_weights(model, weights_path, device, dict_key="state_dict"):
    weights = torch.load(weights_path, map_location=device)[dict_key]
    model.load_state_dict(weights)
    return model

def set_parameter_requires_grad(model, trainable=False, trainable_layer_name=None):
    """Determine which layers should be trainable"""
    # Freeze model layers
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        if trainable_layer_name is not None:
            for name, param in model.named_parameters():
                if not any(i.casefold() in name for i in trainable_layer_name):
                    param.requires_grad = False
            return model
        else:
            for param in model.parameters():
                param.requires_grad = True
            return model

def find_parameter_requires_grad(model):
    """Find which parameters require gradients in order to 
    pass them to the optimizer"""
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("Training: here", name)
    return params_to_update

# def watershed_from_edges(edges, threshold=0.5, erosion_footprint=skimage.morphology.ball(1)):
#     """Perform watershed from an edge map.
    
#     Seed objects are calculated from regional minima in the edge map. These seeds are then eroded
    
#     Resulting watershed segmentation will require some post-processing, like background removal."""

#     th_edges = edges > threshold

#     dist = scipy.ndimage.distance_transform_edt(th_edges)

#     seeds = mahotas.regmin(dist)

#     seeds = skimage.morphology.erosion(seeds, erosion_footprint)

#     seeds = scipy.ndimage.label(seeds)[0]

#     ws = skimage.segmentation.watershed(th_edges, seeds)

#     return ws

def filter_patch(patch, ignore_threshold=0.98, ignore_label=0):
    """Filter patches that contain over the ignore_threshold percentage of ignore labels (eg background)"""
    # Find the number of pixels to ignore
    ignore_count = np.sum(patch == ignore_label)
    # Calculate the percentage of ignore labels
    ignore_perc = ignore_count / patch.size
    print(ignore_threshold, ignore_perc)
    return ignore_threshold > ignore_perc

def create_patch_dataset(load_data_csv, patch_size, filter_patches=False, create_wmap=False, w0=1, sigma=5):
    patch_iter = monai.data.PatchIter(patch_size, mode="reflect")
    if create_wmap:
        columns = ["image", "mask", "weight_map"]
    else:
        columns = ["image", "mask"]
    output_data_csv = pd.DataFrame(columns=columns)
    os.makedirs("./patch_images", exist_ok=True), os.makedirs("./patch_masks", exist_ok=True)
    if create_wmap:
        os.makedirs("./patch_weight_map", exist_ok=True)

    for i in range(len(load_data_csv)):
        image = skimage.io.imread(load_data_csv.iloc[i, 0])
        mask = skimage.io.imread(load_data_csv.iloc[i, 1])
        mask = make_sequential(mask)
        if mask.ndim == 3:
            mask = np.expand_dims(mask, 0) # Patching requires (C, spatial)
        weight_map = mask.copy() # Will calculate actual weight map later
        print(image.shape, mask.shape, weight_map.shape)
        if create_wmap:
            # weight_map = calculate_binary_weight_map(mask, w0=w0, sigma=sigma)
            for ind, (img, msk, wmp) in enumerate(zip(patch_iter(image), patch_iter(mask), patch_iter(weight_map))):
                if filter_patches:
                    if not filter_patch(msk[0]):
                        img_fp, mask_fp, weight_map_fp = f"./patch_images/image_patch_{i}_{ind}.tiff", f"./patch_masks/mask_patch_{i}_{ind}.tiff", f"./patch_weight_map/weight_map_patch_{i}_{ind}.tiff"
                        skimage.io.imsave(img_fp, img[0], compression=("zlib", 1), check_contrast=False)
                        skimage.io.imsave(mask_fp, msk[0], compression=("zlib", 1), check_contrast=False)
                        wmp = calculate_binary_weight_map(wmp[0], w0=w0, sigma=sigma)
                        skimage.io.imsave(weight_map_fp, wmp, compression=("zlib", 1), check_contrast=False)
                        output_data_csv.loc[len(output_data_csv)] = [img_fp, mask_fp, weight_map_fp]
                    else:
                        continue
                else:
                    img_fp, mask_fp, weight_map_fp = f"./patch_images/image_patch_{i}_{ind}.tiff", f"./patch_masks/mask_patch_{i}_{ind}.tiff", f"./patch_weight_map/weight_map_patch_{i}_{ind}.tiff"
                    skimage.io.imsave(img_fp, img[0], compression=("zlib", 1), check_contrast=False)
                    skimage.io.imsave(mask_fp, msk[0], compression=("zlib", 1), check_contrast=False)
                    wmp = calculate_binary_weight_map(wmp[0], w0=w0, sigma=sigma)
                    skimage.io.imsave(weight_map_fp, wmp, compression=("zlib", 1), check_contrast=False)
                    output_data_csv.loc[len(output_data_csv)] = [img_fp, mask_fp, weight_map_fp]
        else:
            for ind, (img, msk) in enumerate(zip(patch_iter(image), patch_iter(mask))):
                if filter_patches:
                    if not filter_patch(msk[0]):
                        img_fp, mask_fp = f"./patch_images/image_patch_{i}_{ind}.tiff", f"./patch_masks/mask_patch_{i}_{ind}.tiff"
                        skimage.io.imsave(img_fp, img[0], compression=("zlib", 1), check_contrast=False)
                        skimage.io.imsave(mask_fp, msk[0], compression=("zlib", 1), check_contrast=False)
                        output_data_csv.loc[len(output_data_csv)] = [img_fp, mask_fp]
                    else:
                        continue
                else:
                    img_fp, mask_fp = f"./patch_images/image_patch_{i}_{ind}.tiff", f"./patch_masks/mask_patch_{i}_{ind}.tiff"
                    skimage.io.imsave(img_fp, img[0], compression=("zlib", 1), check_contrast=False)
                    skimage.io.imsave(mask_fp, msk[0], compression=("zlib", 1), check_contrast=False)
                    output_data_csv.loc[len(output_data_csv)] = [img_fp, mask_fp]
    output_data_csv.to_csv("training_data.csv", index=False)


def filter_objects(labels, max_size=300, min_size=20):
    df = skimage.measure.regionprops_table(labels, properties=("label","axis_major_length"))
    df = pd.DataFrame.from_dict(df)
    min_objects = df[(df["axis_major_length"] < min_size)]["label"].values
    max_objects = df[(df["axis_major_length"] > max_size)]["label"].values
    remove_labels = np.concatenate((min_objects, max_objects), axis=0)
    for rmv in remove_labels:
        labels = np.where(labels == rmv, 0, labels)
    labels = make_sequential(labels)
    return labels

def filter_objects_binary(labels, prob_binary=None, prob_threshold=0.5):
    if prob_binary is None:
        raise ValueError("prob_binary must be provided")

    df = skimage.measure.regionprops_table(
        labels,
        intensity_image=prob_binary,
        properties=("label", "axis_major_length", "intensity_mean")
    )
    df = pd.DataFrame.from_dict(df)

        # filter by intensity
    condition = df["intensity_mean"] <= prob_threshold

    remove_labels = df[condition]["label"].values
    for rmv in remove_labels:
        labels = np.where(labels == rmv, 0, labels)

    labels = make_sequential(labels) 
    return labels

def merge_small_fragments(label_image, min_smallfragment_area=50000, min_largercell_area=300000):
    """
    Merge small label fragments into neighboring cells with a sufficiently large major axis length.

    Parameters:
    - label_image: 3D NumPy array of labeled regions.
    - min_smallfragment_area: minimum length of the major axis to be considered a fragment.
    - min_largercell_area: minimum length of the major axis required for a neighbor to absorb a fragment.

    Returns:
    - A modified label image with small fragments merged into larger adjacent cells.
    """
    # Measure region properties for all labels
    props = regionprops(label_image)

    # Create dictionaries for volume and major axis length per label
    label_area = {p.label: p.area for p in props}

    # Identify labels that are considered small fragments
    small_labels = [lbl for lbl, size in label_area.items() if size < min_smallfragment_area]
    large_labels = set(lbl for lbl, length in label_area.items() if length >= min_largercell_area)

    #print(f"Found {len(small_labels)} small fragments.")

    # Create a copy of the label image to modify
    new_labels = label_image.copy()
    
    # 3D connectivity structure (26 neighbors)
    struct = generate_binary_structure(rank=3, connectivity=3)

    for lbl in small_labels:
        # Create a binary mask for the current small fragment
        mask = label_image == lbl

        # Dilate the mask to find adjacent neighbors
        dilated = binary_dilation(mask)
        neighbors = np.unique(label_image[dilated & (label_image != lbl)])

        if len(neighbors) == 0:
            continue

        # Filter neighbors by major axis length threshold
        large_neighbors = [n for n in neighbors if n in large_labels]

        if len(large_neighbors) == 0:
            continue  # No eligible neighbors found
        
        # Calculate contact area with each large neighbor
        contact_areas = {}
        for neighbor in large_neighbors:
            neighbor_mask = label_image == neighbor
            # Border voxels of fragment touching neighbor voxels
            touching_voxels = binary_dilation(neighbor_mask, structure=struct) & mask
            contact_areas[neighbor] = np.sum(touching_voxels)

        # Select neighbor with maximum contact area
        best_neighbor = max(contact_areas, key=contact_areas.get)

        # Assign the fragment voxels to the selected neighboring label
        new_labels[mask] = best_neighbor

    return new_labels

def intersection_over_union(ground_truth, prediction):
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU

