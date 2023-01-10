from torch.utils.data import Dataset
import skimage
import numpy as np
import torch
import sys
import pandas as pd
sys.path.append("../")
from unet.augmentations.augmentations import (
    Compose,
    ToTensor,
    LabelsToEdgesAndCentroids,
    RandomContrastBrightness,
    RandomGuassianBlur,
    RandomGaussianNoise,
    RandomRotate2D,
    Flip,
    RandomRot90,
    RandomPoissonNoise,
    ElasticDeform,
    RandomScale,
    Normalize
)

class RandomData(Dataset):
    """
    The goal of this class is to generate some small,
    random images for testing training. Thus, you can
    test locally on a CPU before running elsewhere. 
    """
    def __init__(self, dataset_size:int, data_shape:tuple, num_classes:int, train_val="train"):
            self.data_shape = data_shape
            self.dataset_size = dataset_size
            self.num_classes = num_classes
            self.X, self.y = self.build_data(self.dataset_size, self.data_shape, self.num_classes)
            self.train_val = train_val
            self.transforms = {
                "train": 
                    Compose(
                        [
                            RandomContrastBrightness(p=0.5),
                            Flip(p=0.5),
                            RandomRot90(p=0.5),
                            RandomGuassianBlur(p=0.5),
                            RandomGaussianNoise(p=0.5),
                            RandomPoissonNoise(p=0.5),
                            # ElasticDeform(sigma=5, points=1, p=0.5), 
                            Normalize(),
                            ToTensor()
                        ]
                    ),
                "val": 
                    Compose(
                        [
                            Normalize(),
                            ToTensor()
                        ]
                    )
            }

    def __len__(self):
        return self.dataset_size

    def build_data(self, data_size: int, data_shape: tuple, num_classes: int):
        assert len(data_shape) == 4, "data_shape must have shape (channels, depth, height, width)"
        channels, depth, height, width = data_shape
        torch.manual_seed(42)
        data_X = []
        data_y = []
        for i in range(data_size):
            # dtype float32 for input pixels
            data_X.append(
                np.random.rand(channels, depth, height, width).astype(np.float32)
            )
            # dtype int64 since class predictions are ints
            # max = 2 since randint requires it to be one above max. Unsure why
            data_y.append(
                np.random.randint(0, 2, (num_classes, depth, height, width)).astype(np.int64)
            )
        print(data_X[1].shape, data_y[1].shape)
        return data_X, data_y

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.train_val == "train":
            sample = {'image': X, 'mask': y}
            data = self.transforms["train"](**sample)
            return data
        elif self.train_val == "val":
            sample = {'image': X, 'mask': y}
            data = self.transforms["train"](**sample)
            return data
        else:
            raise NotImplementedError
    
    def collate_fn(self, data):
        """Stack images and masks separately into batches
        (batch, classes, D, H, W)"""
        images = []
        masks = []
        for batch in data:
            image = batch["image"]
            mask = batch["mask"]
            images.append(image)
            masks.append(mask)

        images = torch.stack(images, axis=0)
        masks = torch.stack(masks, axis=0)
        return images, masks



class MaddoxDataset(Dataset):
    def __init__(self, data_csv, train_val="train"):
        self.data = data_csv
        self.train_val = train_val
        self.transforms = {
            "train": 
                Compose(
                    [
                        RandomContrastBrightness(p=0.5),
                        Flip(p=0.5),
                        RandomRot90(p=0.5),
                        RandomGuassianBlur(p=0.5),
                        RandomGaussianNoise(p=0.5),
                        RandomPoissonNoise(p=0.5),
                        ElasticDeform(sigma=5, points=1, p=0.5),
                        LabelsToEdgesAndCentroids(centroid_pad=2),
                        Normalize(),
                        ToTensor()
                    ]
                ),
            "val": 
                Compose(
                    [
                        Normalize(),
                        ToTensor()
                    ]
                )
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = skimage.io.imread(self.data.iloc[idx, 0]).astype(np.float32)
        mask = skimage.io.imread(self.data.iloc[idx, 1]).astype(np.float32)
        if self.train_val == "train":
            sample = {'image': image, 'mask': mask}
            data = self.transforms["train"](**sample)
            return data
        elif self.train_val == "val":
            sample = {'image': image, 'mask': mask}
            data = self.transforms["train"](**sample)
            return data
        else:
            raise NotImplementedError

    def collate_fn(self, data):
        """Stack images and masks separately into batches
        (batch, classes, D, H, W)"""
        images = []
        masks = []
        for batch in data:
            image = batch["image"]
            mask = batch["mask"]
            images.append(image)
            masks.append(mask)

        images = torch.stack(images, axis=0)
        masks = torch.stack(masks, axis=0)
        return images, masks

        

