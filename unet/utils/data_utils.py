import numpy as np
import skimage
import torch
import pathlib
import torch.nn.functional as F
import os

def generate_patches(image, patch_shape, stride_shape):
    """Uses PyTorch unfolde to generate non-overlapping patches
    for a given input in 3D.
    
    Patch shape and stride shape are in order (D, W, H). 

    Output tensor with order: (patch, D, W, H).
    """
    # For now, patch and stride shape are the same.
    # stride_shape = patch_shape

    if not torch.is_tensor(image):
        raise TypeError("Input is not a Tensor.")

    # Check that the image dimensions divide cleanly into the patch shape
    # If not, pad the image. 
    if any([
        image.shape[0] % patch_shape[0],
        image.shape[1] % patch_shape[1],
        image.shape[2] % patch_shape[2]]):
        print("Patches do not divide by the image shape. Padding image.")
        image = F.pad(
            image,
            (image.size(2)%patch_shape[2] // 2, image.size(2)%patch_shape[2] // 2,
            image.size(1)%patch_shape[1] // 2, image.size(1)%patch_shape[1] // 2,
            image.size(0)%patch_shape[0] // 2, image.size(0)%patch_shape[0] // 2)
            )
    # Add an extra dimension that will hold the patches
    image = torch.unsqueeze(image, axis=0)
    # Unfold the 1st dimension with size patch_shape[0] with stride_shape[0]
    # Unfold slides along in the provided dimension providing the desired patches
    patches = image.unfold(
        1, patch_shape[0], stride_shape[0]
        ).unfold(
            2, patch_shape[1], stride_shape[1]
            ).unfold(
                3, patch_shape[2], stride_shape[2]
                )
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, patch_shape[0], patch_shape[1], patch_shape[2]) 
    return patches

def save_patches(patches, save_filename, save_dir):
    """
    Takes Tensors with shape (patch, D, W, H) and saves them as 
    .tiff files in save_dir"""
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    patches = patches.numpy()

    all_paths = []
    
    for i in range(patches.shape[0]):
        save_path = pathlib.Path(save_filename)
        out_filename = save_path.with_name(f"{save_path.stem}_patch{i+1}{save_path.suffix}")
        out_path = os.path.join(save_dir, out_filename)
        skimage.io.imsave(out_path, patches[i,...], compression=('zlib', 1))
        all_paths.append(out_path)
    return all_paths

