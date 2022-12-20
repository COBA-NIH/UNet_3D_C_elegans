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

        

