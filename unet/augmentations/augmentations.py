import torch
import skimage
import numpy as np
import random
import scipy
import elasticdeform
from collections.abc import Iterable
from . import aug_functional as F


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
        # Probability for all of compose
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
            if self.paired and (["weight_map"] not in targets):
                print(targets)
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


class LabelsToEdges(DualTransform):
    def __init__(self, mode="thick", connectivity=2, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.mode = mode
        self.connectivity = connectivity
    
    def apply(self, image):
        print("running apply",image.shape)
        return image

    def apply_to_mask(self, mask):
        edges = skimage.segmentation.find_boundaries(mask, mode=self.mode, connectivity=self.connectivity)
        print("running apply to mask",edges.shape)
        return edges

    def apply_to_wmap(self, wmap):
        print("running apply to wmap",wmap.shape)
        return wmap

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
        return F.labels_to_edges_and_centroids(
            mask, self.mode, self.connectivity, self.blur, self.centroid_pad
        )

    def apply_to_wmap(self, wmap):
        return wmap

class LabelsToEdgesAndBinary(DualTransform):
    def __init__(self, mode="thick", connectivity=2, blur=2, p=1):
        super().__init__(p=p)
        self.mode = mode
        self.connectivity = connectivity
        self.blur = blur

    def apply(self, image):
        """The image is not changed"""
        return image

    def apply_to_mask(self, mask):
        return F.labels_to_edges_and_binary(
            mask, self.mode, self.connectivity, self.blur
        )

    def apply_to_wmap(self, wmap):
        return wmap   
    
class ToTensor(DualTransform):
    """Convert input into a tensor. If input has ndim=3 (D, H, W), will expand dims
    to add channel (C, D, H, W)"""

    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, image):
        """The image is not changed"""
        return F.convert_to_tensor(image)


class RandomGuassianBlur(DualTransform):
    """Apply a random sigma of Guassian blur to an array.
    sigma_range controls the extend of the sigma"""

    def __init__(self, sigma_range=[0.1, 2.0], p=1):
        super().__init__(p=p)
        self.sigma_range = sigma_range

    def apply(self, image):
        return F.gaussian_blur(image, self.sigma_range)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap



class RandomGaussianNoise(DualTransform):
    """Draw random samples from a Gaussian distribution
    and apply this as noise to the original image"""

    def __init__(self, scale=[0, 1], p=1):
        super().__init__(p=p)
        self.scale = scale

    def apply(self, image):
        return F.random_gaussian_noise(image, self.scale)

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


class Normalize(DualTransform):
    """Z-score normalization.
    Normalizes the input image so that the mean is 0 and std is 1.
    Can also normalize per channel for tensors with shape (C, spatial)
    """
    def __init__(self, always_apply=True, per_channel=False):
        super().__init__(always_apply)
        self.per_channel = per_channel

    def apply(self, image):
        return F.normalize_img(image, self.per_channel)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap


class RandomContrastBrightness(DualTransform):
    def __init__(self, alpha=1, beta=0, p=1):
        super().__init__(p=p)

    def apply(self, image):
        return F.random_brightness_contrast(image)

    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap

class SelectChannel(DualTransform):
    """Select specific channels to take forward
    Allows for easy testing of which channels are 
    the most influential for a successful segmentation
    """
    def __init__(self, channel=0, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.channel = channel

    def apply(self, image):
        # Select the desired channel and add the channel dim back 
        image = image[self.channel,...]
        image = np.expand_dims(image, 0)
        return image
    
    def apply_to_mask(self, mask):
        return mask

    def apply_to_wmap(self, wmap):
        return wmap

class RandomRotate2D(DualTransform):
    """Rotate a 3D image in axes (W, H)(1, 2)"""

    def __init__(self, angle=30, p=1):
        super().__init__(p=p)
        self.angle = angle
        self.axes = (-2, -1)  # Width and height

    def get_params(self, **data):
        return {"angle": np.random.randint(-self.angle, self.angle)}

    def apply(self, image, angle):
        return scipy.ndimage.rotate(image, axes=self.axes, angle=angle, order=1)

    def apply_to_mask(self, mask, angle):
        return scipy.ndimage.rotate(mask, axes=self.axes, angle=angle, order=0)

    def apply_to_wmap(self, wmap, angle):
        """One-hot encoded has shape (channels, spatial), so we rotate on
        axes 2, 3"""
        return scipy.ndimage.rotate(wmap, axes=self.axes, angle=angle, order=0)


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
    def __init__(self, p=1.0, channel_axis=None):
        super().__init__(p=p, channel_axis=channel_axis)
        self.axis = (-2, -1)

    def get_params(self, **data):
        return {"rotations": np.random.randint(0, 4)}

    def apply(self, image, rotations):
        return np.rot90(image, rotations, axes=self.axis)

    def apply_to_mask(self, mask, rotations):
        return np.rot90(mask, rotations, axes=self.axis)

    def apply_to_wmap(self, wmap, rotations):
        return np.rot90(wmap, rotations, axes=self.axis)


class ElasticDeform(DualTransform):
    def __init__(
        self,
        sigma=25,
        points=3,
        mode="mirror",
        axis=(2, 3),
        p=1.0,
        channel_axis=None,
    ):
        # paired=True since we will use both images and masks in the same self.apply call
        super().__init__(p=p, paired=True)
        self.sigma = sigma
        self.points = points
        self.axis = axis  # Axis on which to apply deformation (default skips z at 0th)
        self.mode = mode
        self.channel_axis = channel_axis

    def apply(self, image, mask, weight_map=None):
        """
        deform_random_grid will perform paired augmentations
        to all arrays within a list. Multichannel arrays
        must be split into single channels and passed, so that
        all input to deform_random_grid is the same
        
        
        pseucode:
        convert arrays to ch, z, x, y

        """
        assert image.ndim == 4, "Input image should be shape (C, spatial)"
        assert mask.ndim == 4, "Input mask should be shape (C, spatial)"
        if weight_map is not None:
            assert weight_map.ndim == 4, "Input weight map should be shape (C, spatial)"
            num_channels = [arr.shape[0] for arr in [image, mask, weight_map]]
            split_arrays = [np.split(arr, arr.shape[0], axis=0) for arr in [image, mask, weight_map]]
        else:
            num_channels = [arr.shape[0] for arr in [image, mask]]
            split_arrays = [np.split(arr, arr.shape[0], axis=0) for arr in [image, mask]]
    
        # Flatten list to list of arrays
        data = [arr for split_arr in split_arrays for arr in split_arr]

        # Iterpolate only the raw image pixels
        interpolate_order = np.zeros(len(data), dtype=int)
        interpolate_order[0:num_channels[0]] = 1
        interpolate_order = list(interpolate_order)
        # Perform elasticdeformation
        data = elasticdeform.deform_random_grid(
            data,
            sigma=self.sigma,
            points=self.points,
            axis=self.axis,
            order=interpolate_order,
            mode=self.mode,
        )

        image = np.concatenate(data[0:num_channels[0]], axis=0)
        mask = np.concatenate(data[num_channels[0]:sum(num_channels[0:2])], axis=0)
        if weight_map is not None:
            weight_map = np.concatenate(data[sum(num_channels[0:2]):sum(num_channels[0:3])], axis=0)
            return image, mask, weight_map
        else:
            return image, mask


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
        return F.edges_and_centroids(
            mask,
            mode=self.mode,
            connectivity=self.connectivity,
            iterations=self.iterations,
        )

    def apply_to_wmap(self, wmap):
        return wmap


class EdgesAndCentroids2(DualTransform):
    def __init__(self, mode="thick", connectivity=1, dilation=1, always_apply=True):
        super().__init__(always_apply)
        self.mode = mode
        self.connectivity = connectivity
        self.dilation = dilation

    def apply(self, image):
        """The image is not changed"""
        return image

    def apply_to_mask(self, mask):
        return F.edges_and_centroids2(
            mask,
            mode=self.mode,
            connectivity=self.connectivity,
            dilation=self.dilation,
        )

    def apply_to_wmap(self, wmap):
        return wmap


class BlurMasks(DualTransform):
    """Apply Gaussian blur to masks only"""

    def __init__(self, sigma=2, blur_axis=None, channel_axis=0, always_apply=True):
        super().__init__(always_apply)
        self.sigma = sigma
        self.blur_axis = blur_axis
        self.channel_axis = channel_axis

    def apply(self, image):
        return image

    def apply_to_mask(self, mask):
        # Blur axis assumes the channel dim is 0th. Not ideal.
        if self.blur_axis is not None:
            if not isinstance(self.blur_axis, Iterable):
                mask[self.blur_axis,...] = skimage.filters.gaussian(
                    mask[self.blur_axis,...], sigma=self.sigma
                    )
                return mask
            else:
                for ax in self.blur_axis:
                    mask[ax,...] = skimage.filters.gaussian(
                        mask[ax,...], sigma=self.sigma
                        )   
                return mask
        else:
            return skimage.filters.gaussian(
                mask, sigma=self.sigma, channel_axis=self.channel_axis
            )

    def apply_to_wmap(self, wmap):
        # return skimage.filters.gaussian(wmap, sigma=self.sigma)
        return wmap

class EdgeMaskWmap(DualTransform):
    def __init__(self, edge_multiplier=1, wmap_multiplier=1, invert_wmap=False, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.edge_multiplier = edge_multiplier
        self.wmap_multiplier = wmap_multiplier
        self.invert_wmap = invert_wmap

    def apply(self, image):
        print("running Edge Mask Wmap apply",image.shape)
        return image
    
    def apply_to_mask(self, mask):
        self.mask = mask
        print("running Edge Mask Wmap apply to mask",mask.shape)
        return mask

    def apply_to_wmap(self, wmap):
        print("running Edge Mask Wmap apply to wmap",wmap.shape)
        wmap = (self.edge_multiplier * self.mask) + (self.wmap_multiplier * wmap) 
        # wmap[self.mask == 0] = 0
        wmap = np.where(self.mask == 0, 0, wmap)
        if self.invert_wmap and self.mask.max() != 0:
            non_zero_mask = wmap != 0
            wmap[non_zero_mask] = np.max(wmap[non_zero_mask]) - wmap[non_zero_mask] + np.min(wmap[non_zero_mask])
        # wmap = self.mask + wmap
        print(wmap.shape)
        return wmap
class BinaryMaskWmap(DualTransform):
    def __init__(self, edge_multiplier=1, wmap_multiplier=1, invert_wmap=False, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.edge_multiplier = edge_multiplier
        self.wmap_multiplier = wmap_multiplier
        self.invert_wmap = invert_wmap

    def apply(self, image):
        return image
    
    def apply_to_mask(self, mask):
        self.mask = mask
        return mask

    def apply_to_wmap(self, wmap):
        print("wmap shape", wmap.shape, "mask shape", self.mask.shape)
        wmap_0 = (self.edge_multiplier * self.mask) + (self.wmap_multiplier * wmap) 
        # wmap[self.mask == 0] = 0
        wmap_0 = np.where(self.mask == 0, 0, wmap_0)
        if self.invert_wmap and self.mask.max() != 0:
            non_zero_mask = wmap_0 != 0
            wmap_0[non_zero_mask] = np.max(wmap_0[non_zero_mask]) - wmap_0[non_zero_mask] + np.min(wmap_0[non_zero_mask])
        # wmap = self.mask + wmap
        print(wmap_0.shape,self.mask.shape)
        return wmap_0
