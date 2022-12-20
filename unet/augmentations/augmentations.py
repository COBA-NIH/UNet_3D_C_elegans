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

    def __init__(self, always_apply=False, p=0.5, paired=False):
        # Ensure the probability is in range
        assert 0 <= p <= 1
        # Determine if the transform is essential
        self.always_apply = always_apply
        self.p = p
        # Controls if augmentations should be applied
        # in the same function call
        self.paired = paired

    def __call__(self, targets, **data):
        # For paired augmentations, transform is never directly called, it's always
        # DualTransform
        if self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                else:
                    data[k] = v

    def get_params(self, **data):
        """get_params is used in the scenario when you are setting
        a random variable that needs to be available to both image and
        mask augmentations"""
        return {}

        return data

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

            if self.paired:
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
            else:
                # Iterate through all of the data
                for k, v in data.items():
                    # If the key is an image, apply it
                    if k in targets[0]:
                        data[k] = self.apply(v, **params)
                    # If the key is a mask, apply that method
                    # Why apply them differently? Well, you don't want resizing of a binary
                    # mask to have interpolation added, do you?
                    elif k in targets[1]:
                        data[k] = self.apply_to_mask(v, **params)
                    else:
                        data[k] = v

        return data

    def apply_to_mask(self, mask, **params):
        """If the augmentation class does not provide its own
        apply_to_mask, just use the apply method instead."""
        return self.apply(mask, **params)


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


def labels_to_edges_and_centroids(labels, mode, connectivity, blur, centroid_pad):
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


def normalize_img(image):
    mean, std = image.mean(), image.std()
    norm_image = (image - mean) / std
    return norm_image


class Normalize(DualTransform):
    """Z-score normalization.
    Normalizes the input image so that the mean is 0 and std is 1."""

    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, image):
        return normalize_img(image)

    def apply_to_mask(self, mask):
        return mask


def random_brightness_contrast(
    input_array, alpha, beta, contrast_limit=0.2, brightness_limit=0.2
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
        return normalize_img(image)

    def apply_to_mask(self, mask):
        return mask


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


class Flip(DualTransform):
    """Select a random set of axis to flip on."""

    def __init__(self, p=1, axis=None):
        super().__init__(p=p)
        self.axis = axis

    def get_params(self, **data):
        if self.axis is not None:
            axis = self.axis
        else:
            axis_combinations = [(1,), (2,), (1, 2)]
            axis = random.choice(axis_combinations)
        return {"axis": axis}

    def apply(self, image, axis):
        return np.flip(image, axis=axis)

    def apply_to_mask(self, mask, axis):
        return np.flip(mask, axis=axis)


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


class RandomRot90(DualTransform):
    def __init__(self, axis=(1, 2), p=1.0):
        super().__init__(p=p)
        self.axis = axis

    def get_params(self, **data):
        return {"rotations": np.random.randint(0, 4)}

    def apply(self, image, rotations):
        return np.rot90(image, rotations, axes=self.axis)

    def apply_to_mask(self, mask, rotations):
        return np.rot90(mask, rotations, axes=self.axis)


class ElasticDeform(DualTransform):
    def __init__(self, paired=True, sigma=5, points=2, p=1.0):
        """Paired controls if the apply method should pass both image and mask
        to the same apply method"""
        super().__init__(p=p, paired=True)
        self.sigma = sigma
        self.points = points

    def apply(self, image, mask):
        image, mask = elasticdeform.deform_random_grid(
            [image, mask], sigma=self.sigma, points=self.points, order=[1, 0]
        )
        return image, mask
