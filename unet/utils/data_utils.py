import numpy as np
import scipy
import skimage
import torch
import pathlib
import torch.nn.functional as F
import os
import mahotas

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
            print("Training:", name)
    return params_to_update

def watershed_from_edges(edges, threshold=0.5, erosion_footprint=skimage.morphology.ball(1)):
    """Perform watershed from an edge map.
    
    Seed objects are calculated from regional minima in the edge map. These seeds are then eroded
    
    Resulting watershed segmentation will require some post-processing, like background removal."""

    th_edges = edges > threshold

    dist = scipy.ndimage.distance_transform_edt(th_edges)

    seeds = mahotas.regmin(dist)

    seeds = skimage.morphology.erosion(seeds, erosion_footprint)

    seeds = scipy.ndimage.label(seeds)[0]

    ws = skimage.segmentation.watershed(th_edges, seeds)

    return ws
