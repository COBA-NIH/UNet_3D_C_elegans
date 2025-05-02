import skimage
import torch
import numpy as np
import random
import scipy

def resize(img, new_shape, interpolation=1):
    new_img = skimage.transform.resize(
        img, new_shape, order=interpolation, anti_aliasing=False
    )
    return new_img


def labels_to_edges_and_centroids(labels, connectivity, blur, centroid_pad):
    """For a given instance labelmap, convert the labels to edges and centroids.
    Blur the edges if you'd like.

    Centroids are rounded to the nearest pixel

    Returns a two channel image with shape (C, Z, H, W).

    Edges are ch0 and centers are ch1"""
    labels = labels.astype(int)
    regions = skimage.measure.regionprops(labels)
    if len(regions) > 0:
        cell_edges = skimage.segmentation.find_boundaries(labels, connectivity)
        cell_edges = skimage.filters.gaussian(cell_edges, sigma=blur)
        centers = np.zeros_like(labels)
        for lab in regions:
            x, y, z = lab.centroid
            x, y, z = round(x), round(y), round(z)
            centers[
                x - centroid_pad : x + centroid_pad,
                y - centroid_pad : y + centroid_pad,
                z - centroid_pad : z + centroid_pad,
            ] = 1
        output = [cell_edges, centers]
    else:
        # GT is blank
        output = [labels, labels]

    # Add background as a class
    background = np.zeros_like(labels)
    background[labels == 0] = 1
    output.insert(0, background)
    return np.stack(output, axis=0)

def convert_to_tensor(input_array):
    assert input_array.ndim in [3, 4], "Image must be 3D (D, H, W) or 4D (C, D, H, W)"
    if input_array.ndim == 3:
        # Add channel axis
        input_array = np.expand_dims(input_array, axis=0)
    return torch.from_numpy(input_array.astype(np.float32))


def gaussian_blur(input_array, sigma_range=[0.1, 2.0]):
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    input_array = scipy.ndimage.gaussian_filter(input_array, sigma)
    return input_array


def random_gaussian_noise(input_array, scale=[0, 1]):
    std = np.random.uniform(scale[0], scale[1])
    noise = np.random.normal(0, std, input_array.shape)
    return input_array + noise


def normalize_img(image, per_channel=False):
    # mean, std = image.mean(), image.std()
    # norm_image = (image - mean) / std
    # return norm_image

    if per_channel:
        # Get the axes to compute mean for
        axes = tuple(list(range(image.ndim)))[1:]
        mean = np.mean(image, axis=axes, keepdims=True)
        std = np.std(image, axis=axes, keepdims=True)
    else:
        mean = np.mean(image)
        std = np.std(image)

    return (image - mean) / np.clip(std, a_min=1e-10, a_max=None)



def random_brightness_contrast(
    input_array, alpha=1.0, beta=0.0, contrast_limit=0.2, brightness_limit=0.2
):
    alpha = alpha + np.random.uniform(-contrast_limit, contrast_limit)
    beta = beta + np.random.uniform(-brightness_limit, brightness_limit)
    input_array = (alpha * input_array)
    input_array += beta * input_array.mean()
    return input_array



def edges_and_centroids(
    labels,
    connectivity=1,
    mode="inner",
    return_initial_border=True,
    iterations=1,
    one_hot=True,
):
    """Calculate the border around objects and then subtract this border from
    the object, reducing the object size"""
    assert iterations > 0, "Iterations must be greater than 0"
    centroids = labels.copy().astype(int)
    for i in range(iterations):
        # Calculate the edges. We use centroids since we will erode them over iterations
        edges = skimage.segmentation.find_boundaries(
            centroids, connectivity=connectivity, mode=mode
        )
        # If a pixel was determined to be a boundary of an object, remove it
        centroids[edges > 0.5] = 0
        # Make labels binary
        centroids[centroids > 0.5] = 1
    if return_initial_border:
        edges = skimage.segmentation.find_boundaries(
            labels, connectivity=connectivity, mode=mode
        )

    if one_hot:
        output = [edges, centroids]
        background = np.zeros_like(labels)
        # Background is where there is no foreground
        background[labels == 0] = 1
        # Make background the 0th index
        output.insert(0, background)
        return np.stack(output, axis=0)
    else:
        # Bump centroid label up to 2
        centroids[centroids == 1] = 2
        output = np.stack([edges, centroids], axis=0)
        return np.max(output, axis=0)


def edges_and_centroids2(
    labels,
    mode,
    connectivity=1,
    dilation=1,
    one_hot=True
    ):
    """Centroids are calculated using skimage regionprops and then 
    dilated. Gives a more uniform centroid shape, independent of GT size"""
    """Calculate the border around objects and then subtract this border from
    the object, reducing the object size"""
    label_properties = skimage.measure.regionprops(labels.astype(int))
    # Gather centroids
    centroids = [prop.centroid for prop in label_properties]
    # Round centroids
    centroids = np.round(centroids).astype(int)
    # Create a image to add the centroids to
    centroid_image = np.zeros_like(labels)
    # Add centroids 
    # .T transpose will make the centroid array have shape
    # (dimension, num_centroids) ie. for a 3D image (3, X)
    centroid_image[tuple(centroids.T)] = 1
    # Dilate centroid markers
    centroid_image = skimage.morphology.dilation(centroid_image, skimage.morphology.ball(dilation))

    edges = skimage.segmentation.find_boundaries(
            labels, connectivity=connectivity, mode=mode
        )

    if one_hot:
        background = np.zeros_like(labels)
        # Background is where there is no GT
        background[labels == 0] = 1
        output = [background, edges, centroid_image]
        return np.stack(output, axis=0)
    else:
        # Bump centroid label up to 2
        centroid_image[centroid_image == 1] = 2
        output = np.stack([edges, centroid_image], axis=0)
        return np.max(output, axis=0)

def labels_to_edges_and_binary(labels, mode, connectivity, blur):

    labels = labels.astype(int)
    print(f'labels shape: {labels.shape}')
    labels_squeeze = np.squeeze(labels)
    regions = skimage.measure.regionprops(labels_squeeze)
    #labels= np.expand_dims(output, axis=0)
    if len(regions) > 0:
        print("this label patch is not blank")
        cell_edges = skimage.segmentation.find_boundaries(labels_squeeze, connectivity)
        cell_edges = skimage.filters.gaussian(cell_edges, sigma=blur)
        print(f'cell_edges shape: {cell_edges.shape}')
        foreground = np.zeros_like(labels_squeeze)
        foreground[labels_squeeze >= 1] = 1
        print(f'foreground shape: {foreground.shape}')
        output = [cell_edges, foreground]        
    else:
        print("this label patch is blank")
        # GT is blank
        output = [labels_squeeze, labels_squeeze]
    output = np.stack(output, axis=0)
    #output_4d = np.expand_dims(output, axis=0)
    print("labels function",labels_squeeze.shape,output.shape)
    return output[1:2] #just get the foreground as a mask