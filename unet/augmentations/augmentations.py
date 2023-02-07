import torch
import skimage
import numpy as np
import random
import scipy
import elasticdeform


class Compose:
    def __init__(self, transforms, p=1.0, targets=[["image"], ["mask"]]):
        """When Compose is initialized, look at the transforms it contains (a list)
        and what the probability is for the entire transform"""
        assert 0 <= p <= 1
        self.transforms = transforms  # + [Contiguous(always_apply=True)] # Ensure outputs are always contiguous
        self.p = p
        self.targets = targets
        # assert self.targets in ["image", "mask", "wmap"], f"Expected targets to be in ['image', 'mask', 'wmap'], got {self.targets}."

    def get_always_apply_transforms(self):
        """If a transformation is to be always applied, find them."""
        res = []
        for tr in self.transforms:
            if tr.always_apply:
                res.append(tr)
        return res

    def __call__(self, **data):
        """When Compose is actually called (ie. when passed data), check the Compose probability
        and then apply each transform in the list if True"""
        need_to_run = random.random() < self.p
        transforms_to_apply = (
            self.transforms if need_to_run else self.get_always_apply_transforms()
        )
        for tr in transforms_to_apply:
            data = tr(self.targets, **data)

        return data


class Transform:
    """Base transformation class. Mostly changed with super().__init__()"""

    def __init__(self, always_apply=False, p=0.5, paired=False, channel_axis=None):
        # Ensure the probability is in range
        assert 0 <= p <= 1
        # Determine if the transform is essential
        self.always_apply = always_apply
        self.p = p
        # Controls if augmentations should be applied
        # in the same function call
        self.paired = paired
        self.channel_axis = channel_axis

    def __call__(self, targets, **data):
        # For paired augmentations, transform is never directly called, it's always
        # DualTransform
        if self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            for k, v in data.items():
                if self.channel_axis is not None:
                    # If there is a channel dimension, split channels into 
                    # a list of arrays
                    v = list(np.swapaxes(v, 0, self.channel_axis))
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                else:
                    data[k] = v

    def get_params(self, **data):
        """get_params is used in the scenario when you are setting
        a random variable that needs to be available to both image and
        mask augmentations"""
        return {}

    def apply(self, volume, **params):
        """Shouldn't end up here for transformations. If you do,
        it's likely because your augmentation class doesn't have an apply
        method or it hasn't been super()'d"""
        raise NotImplementedError


class DualTransform(Transform):
    """Class to handle paired transformation of data, as defined by "targets" in Compose"""

    def __call__(self, targets, **data):
        # When called, check if this individual transform has to be force applied
        # (defined at the transform level), always applied (defined at the Compose level)
        # or check the probability that it's applied.
        if self.always_apply or random.random() < self.p:
            params = self.get_params(**data)
            # self.paired == True means that image and mask
            # will have the same transforms applied equally
            if self.paired and ["weight_map"] not in targets:
                image, mask = self.apply(**data)
                # Add paired transforms back into the expected
                # dictionary keys
                for k, v in data.items():
                    if k in targets[0]:
                        data[k] = image
                    elif k in targets[1]:
                        data[k] = mask
                    else:
                        raise NotImplementedError
            # Case where there is a weight map provided
            elif self.paired and ["weight_map"] in targets:
                image, mask, wmap = self.apply(**data)
                # Add paired transforms back into the expected
                # dictionary keys
                for k, v in data.items():
                    if k in targets[0]:
                        data[k] = image
                    elif k in targets[1]:
                        data[k] = mask
                    elif k in targets[2]:
                        data[k] = wmap
                    else:
                        raise NotImplementedError
            else:
                # Iterate through all of the data
                for k, v in data.items():
                    # If the key is an image, apply it
                    if k in targets[0]:
                        if self.channel_axis is not None:
                            # If there is a channel dimension, split channels into 
                            # a list of arrays
                            v = list(np.swapaxes(v, 0, self.channel_axis))
                            v = self.apply(v, **params)
                            data[k] = np.stack(v, axis=self.channel_axis)
                        else:
                            data[k] = self.apply(v, **params)
                    # If the key is a mask, apply that method
                    # Why apply them differently? Well, you don't want resizing of a binary
                    # mask to have interpolation added, do you?
                    elif k in targets[1]:
                        data[k] = self.apply_to_mask(v, **params)
                    elif k in targets[2]:
                        # Treat weight maps like a mask in terms of transformation
                        data[k] = self.apply_to_wmap(v, **params)
                    else:
                        data[k] = v

        return data

    def apply_to_mask(self, mask, **params):
        """If the augmentation class does not provide its own
        apply_to_mask, just use the apply method instead."""
        return self.apply(mask, **params)

    def apply_to_wmap(self, wmap, **params):
        """Most augmentations to the wmap are the same that are applied to the 
        mask. However, having an additional apply allows for the augmentations
        to deviate."""
        return self.apply(wmap, **params)


class Contiguous(DualTransform):
    def apply(self, image):
        return np.ascontiguousarray(image)


def resize(img, new_shape, interpolation=1):

    new_img = skimage.transform.resize(
        img, new_shape, order=interpolation, anti_aliasing=False
    )
    return new_img


class Resize(DualTransform):
    """Class to handle passing of data to resize function"""

    def __init__(self, shape, interpolation=1, resize_type=0, p=1):
        # On init, pass the probability to Transform
        super().__init__(p=p)
        self.shape = shape
        self.interpolation = interpolation
        self.resize_type = resize_type

    def apply(self, img):
        """Resize the image"""
        return resize(img, new_shape=self.shape, interpolation=self.interpolation)

    def apply_to_mask(self, mask):
        """Resize the mask, but don't apply interpolation"""
        return resize(mask, new_shape=self.shape, interpolation=0)

    def apply_to_wmap(self, wmap):
        """Resize the mask, but don't apply interpolation"""
        return resize(wmap, new_shape=self.shape, interpolation=0)


class LabelsToEdgesAndCentroids(DualTransform):
    def __init__(self, mode="thick", connectivity=2, blur=2, centroid_pad=2, p=1):
        super().__init__(p=p)
        self.mode = mode
        self.connectivity = connectivity
        self.blur = blur
        self.centroid_pad = centroid_pad

    def apply(self, image):
        """The image is not changed"""
        return image

    def apply_to_mask(self, mask):
        return labels_to_edges_and_centroids(
            mask, self.mode, self.connectivity, self.blur, self.centroid_pad
        )

    def apply_to_wmap(self, wmap):
        return wmap


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


class ToTensor(DualTransform):
    """Convert input into a tensor. If input has ndim=3 (D, H, W), will expand dims
    to add channel (C, D, H, W)"""

    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, image):
        """The image is not changed"""
        return convert_to_tensor(image)


def convert_to_tensor(input_array):
    assert input_array.ndim in [3, 4], "Image must be 3D (D, H, W) or 4D (C, D, H, W)"
    if input_array.ndim == 3:
        # Add channel axis
        input_array = np.expand_dims(input_array, axis=0)
    return torch.from_numpy(input_array.astype(np.float32))
    #     tensor = torch.from_numpy(input_array).double()
    # return tensor


def gaussian_blur(input_array, sigma_range=[0.1, 2.0]):
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    input_array = scipy.ndimage.gaussian_filter(input_array, sigma)
    return input_array


class RandomGuassianBlur(DualTransform):
    """Apply a random sigma of Guassian blur to an array.
    sigma_range controls the extend of the sigma"""

    def __init__(self, sigma_range=[0.1, 2.0], p=1):
        super().__init__(p=p)
        self.sigma_range = sigma_range

    def apply(self, image):
        return gaussian_blur(image, self.sigma_range)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap     


def random_gaussian_noise(input_array, scale=[0, 1]):
    std = np.random.uniform(scale[0], scale[1])
    noise = np.random.normal(0, std, input_array.shape)
    return input_array + noise


class RandomGaussianNoise(DualTransform):
    """Draw random samples from a Gaussian distribution
    and apply this as noise to the original image"""

    def __init__(self, scale=[0, 1], p=1):
        super().__init__(p=p)
        self.scale = scale

    def apply(self, image):
        return random_gaussian_noise(image, self.scale)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap


class RandomPoissonNoise(DualTransform):
    def __init__(self, lam=[0.0, 1.0], p=1):
        super().__init__(p=p)
        self.lam = lam

    def apply(self, image):
        lam = np.random.uniform(self.lam[0], self.lam[1])
        noise = np.random.poisson(lam, image.shape)
        return image + noise

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap


def normalize_img(image):
    # mean, std = image.mean(), image.std()
    # norm_image = (image - mean) / std
    # return norm_image

    mean = np.mean(image)
    std = np.std(image)

    return (image - mean) / np.clip(std, a_min=1e-10, a_max=None)


class Normalize(DualTransform):
    """Z-score normalization.
    Normalizes the input image so that the mean is 0 and std is 1."""

    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, image):
        return normalize_img(image)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap


def random_brightness_contrast(
    input_array, alpha=1.0, beta=0.0, contrast_limit=0.2, brightness_limit=0.2
):
    alpha = alpha + np.random.uniform(-contrast_limit, contrast_limit)
    beta = beta + np.random.uniform(-brightness_limit, brightness_limit)
    input_array *= alpha
    input_array += beta * input_array.mean()
    return input_array


class RandomContrastBrightness(DualTransform):
    def __init__(self, alpha=1, beta=0, p=1):
        super().__init__(p=p)

    def apply(self, image):
        return random_brightness_contrast(image)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap


class RandomRotate2D(DualTransform):
    """Rotate a 3D image in axes (W, H)(1, 2)"""

    def __init__(self, angle=30, p=1):
        super().__init__(p=p)
        self.angle = angle
        self.axes = (1, 2)  # Width and height

    def get_params(self, **data):
        return {"angle": np.random.randint(-self.angle, self.angle)}

    def apply(self, image, angle):
        return scipy.ndimage.rotate(image, axes=self.axes, angle=angle, order=1)

    def apply_to_mask(self, mask, angle):
        return scipy.ndimage.rotate(mask, axes=self.axes, angle=angle, order=0)

    def apply_to_wmap(self, wmap, angle):
        """One-hot encoded has shape (channels, spatial), so we rotate on 
        axes 2, 3"""
        return scipy.ndimage.rotate(wmap, axes=(2, 3), angle=angle, order=0)


class Flip(DualTransform):
    """Select a random set of axis to flip on."""

    def __init__(self, p=1, axis=None):
        super().__init__(p=p)
        self.axis = axis

    def get_params(self, **data):
        if self.axis is not None:
            axis = self.axis
        else:
            # axis_combinations = [(1,), (2,), (1, 2)]
            axis_combinations = [(-2,), (-1,), (-2, -1)]
            axis = random.choice(axis_combinations)
        return {"axis": axis}

    def apply(self, image, axis):
        return np.flip(image, axis=axis)

    def apply_to_mask(self, mask, axis):
        return np.flip(mask, axis=axis)

    def apply_to_wmap(self, wmap, axis):
        return np.flip(wmap, axis=axis)


class RandomScale(DualTransform):
    def __init__(self, scale_limit=[0.9, 1.1], p=1.0):
        super().__init__(p=p)
        self.scale_limit = scale_limit

    def get_params(self, **data):
        """Make sure both applies have access to random variable"""
        return {"scale": np.random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, image, scale):
        return skimage.transform.rescale(image, scale, order=1)

    def apply_to_mask(self, mask, scale):
        return skimage.transform.rescale(mask, scale, order=0)

    def apply_to_wmap(self, wmap, scale):
        return skimage.transform.rescale(wmap, scale, order=0)


class RandomRot90(DualTransform):
    def __init__(self, axis=(1, 2), p=1.0, channel_axis=None):
        super().__init__(p=p, channel_axis=channel_axis)
        self.axis = axis

    def get_params(self, **data):
        return {"rotations": np.random.randint(0, 4)}

    def apply(self, image, rotations):
        if isinstance(image, list):
            for i, ch_img in enumerate(image):
                print(i)
                image[i] = np.rot90(ch_img, rotations, axes=self.axis)
            return image
        else:
            return np.rot90(image, rotations, axes=self.axis)

    def apply_to_mask(self, mask, rotations):
        return np.rot90(mask, rotations, axes=self.axis)

    def apply_to_wmap(self, wmap, rotations):
        """One-hot encoded has shape (channels, spatial), so we rotate on 
        axes 2, 3"""
        return np.rot90(wmap, rotations, axes=(2, 3))


class ElasticDeform(DualTransform):
    def __init__(self, sigma=25, points=3, mode="constant", axis=(1, 2), p=1.0, channel_axis=None):
        """Paired controls if the apply method should pass both image and mask
        to the same apply method"""
        super().__init__(p=p, channel_axis=channel_axis)
        self.sigma = sigma
        self.points = points
        self.axis = axis # Axis on which to apply deformation (skip z)
        self.mode = mode
        self.channel_axis = channel_axis

    def apply_to_all(self, image, mask, wmap=None):
        """Convert 4D (multiple channel) input images or wmap
        into a flattened list of 3D arrays for elasticdeform"""
        # Detect how many channels
        num_channels = image.shape[self.channel_axis]

        # Flatten into a list
        ch_imgs = [image[i,...] for i in range(num_channels)]
        data = [image, mask]
        data.extend(ch_imgs)

        # Iterpolate only the raw pixels
        interpolate_order = np.zeros(len(data))
        interpolate_order[0:num_channels] = 1

        # Perform elasticdeformation
        data = elasticdeform.deform_random_grid(
            data, 
            sigma=self.sigma, 
            points=self.points, 
            axis=self.axis,
            order=interpolate_order, 
            mode=self.mode,
        )

        # Use slices to stack output
        image = np.stack(data[0:num_channels], axis=0)

        return image, data[1], wmap

    def apply(self, image, mask, weight_map=None):
        if weight_map is not None:
            # wmap is one-hot encoded with shape (channels, spatial)
            # but the shape must be the same for deform_random_grid
            # Split along the channel axis, pass to deform, and then stack
            n_classes = weight_map.shape[0]
            weight_maps = [weight_map[i,...] for i in range(n_classes)]

            data = [image, mask]
            data.extend(weight_maps)

            data = elasticdeform.deform_random_grid(
                data, 
                sigma=self.sigma, 
                points=self.points, 
                axis=self.axis,
                order=[1, 0, 0, 0, 0], # Iterpolate only the raw pixels
                mode=self.mode,
                # mode="constant", # Leads to background
                # mode="nearest", # raw pixels significantly warped
                # mode="wrap", # similar to nearest
                # mode="reflect", # Leads to incomplete edges
                # mode="mirror", # Leads to some strange object shapes
            )

            weight_map = np.stack(data[2:], axis=0)

            return data[0], data[1], weight_map
        else:
            image, mask = elasticdeform.deform_random_grid(
                [image, mask], 
                sigma=self.sigma, 
                points=self.points, 
                axis=self.axis,
                order=[1, 0],
                mode="mirror",
            )
            
            return image, mask


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


class EdgesAndCentroids(DualTransform):
    def __init__(self, mode="inner", connectivity=1, iterations=1, always_apply=True):
        super().__init__(always_apply)
        self.mode = mode
        self.connectivity = connectivity
        self.iterations = iterations

    def apply(self, image):
        """The image is not changed"""
        return image

    def apply_to_mask(self, mask):
        return edges_and_centroids(mask, mode=self.mode, connectivity=self.connectivity, iterations=self.iterations)

    def apply_to_wmap(self, wmap):
        return wmap


class BlurMasks(DualTransform):
    """Apply Gaussian blur to masks only"""
    def __init__(self, sigma=2, channel_axis=0, always_apply=True):
        super().__init__(always_apply)
        self.sigma = sigma
        self.channel_axis = channel_axis
    
    def apply(self, image):
        return image

    def apply_to_mask(self, mask):
        return skimage.filters.gaussian(mask, sigma=self.sigma, channel_axis=self.channel_axis)

    def apply_to_wmap(self, wmap):
        # return skimage.filters.gaussian(wmap, sigma=self.sigma)
        return wmap