# import sys
# sys.path.append("../")
# import os
# import shutil
# import tempfile
# import time
import numpy as np

# from monai.losses import DiceLoss
# from monai.inferers import sliding_window_inference
# import logging
# from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter

# http://localhost:6006

import torch
# import torchvision
# import torch.nn as nn
# import skimage
import pandas as pd
# import sklearn
import sklearn.model_selection

from unet.utils.load_data import MaddoxDataset, RandomData
from unet.networks.unet3d import UNet3D

# from unet.networks.unet3d import UnetModel

from unet.utils.loss import WeightedBCELoss, WeightedBCEDiceLoss
from unet.utils.trainer import RunTraining

import argparse

import unet.augmentations.augmentations as aug


parser = argparse.ArgumentParser(description="3DUnet Training")

# nargs="?" required to fall back to default if no arg provided
parser.add_argument("data", nargs="?", default="patch_data/load_data_training.csv")
parser.add_argument("--batch", nargs="?", default=4, type=int)
parser.add_argument("--epochs", nargs="?", default=10, type=int)
parser.add_argument("--workers", nargs="?", default=4, type=int)
parser.add_argument("--dummy", action="store_true")  # Use dummy data

transforms = {
            "train": 
                aug.Compose(
                    [
                        aug.RandomContrastBrightness(p=0.5),
                        aug.Flip(p=0.5),
                        aug.RandomRot90(p=0.5),
                        aug.RandomGuassianBlur(p=0.5),
                        aug.RandomGaussianNoise(p=0.5),
                        aug.RandomPoissonNoise(p=1),
                        aug.ElasticDeform(p=1),
                        aug.EdgesAndCentroids(),
                        # aug.BlurMasks(sigma=1),
                        aug.Normalize(),
                        aug.ToTensor()
                    ],
                    targets=[["image"], ["mask"], ["weight_map"]]
                ),
            "val": 
                aug.Compose(
                    [
                        aug.EdgesAndCentroids(),
                        aug.Normalize(),
                        aug.ToTensor()
                    ],
                    targets=[["image"], ["mask"], ["weight_map"]]
                )
        }


def main():
    args = parser.parse_args()

    main_worker(args)


def main_worker(args):
    if args.dummy:
        print("----- Using dummy data ------")
        train_ds = RandomData(
            data_shape=(1, 24, 24, 24),
            dataset_size=20,
            num_classes=3,
            train_val="train"
        )
        val_ds = RandomData(
            data_shape=(1, 24, 24, 24), 
            dataset_size=5, 
            num_classes=3, 
            train_val="val"
        )
    else:
        load_csv = pd.read_csv(args.data)
        train_dataset, val_dataset = sklearn.model_selection.train_test_split(
            load_csv, test_size=0.2
        )
        print(
            f"loading data from: {args.data}. Train data of length {train_dataset.shape[0]} and val data of length {val_dataset.shape[0]}"
        )
        train_ds = MaddoxDataset(data_csv=train_dataset, transforms=transforms, train_val="train", wmap=True)

        val_ds = MaddoxDataset(data_csv=val_dataset, transforms=transforms, train_val="val", wmap=True)

    if torch.cuda.is_available():
        # Find fastest conv
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        collate_fn=train_ds.collate_fn,
        num_workers=args.workers,
    )

    # Don't shuffle validation so you can see how predictions improve over time
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        collate_fn=val_ds.collate_fn,
        num_workers=args.workers,
    )

    data_loader = {"train": train_loader, "val": val_loader}

    # model = UNet3D(
    #     input_channels=1, num_classes=3, network_depth=4, activation=None
    # )

    model = UNet3D(
        in_channels=1, out_channels=3, final_sigmoid=True, testing=True
    )

    # Different CUDA, different pytorch handling
    try:
        if torch._C._cuda_getDeviceCount() > 1:
            print("Running on multiple GPUs")
            model = torch.nn.DataParallel(model)
    except:
        if torch.cuda.device_count() > 1:
            print("Running on multiple GPUs")
            model = torch.nn.DataParallel(model)

    model.to(device)

    ## Requries more setup: https://pytorch.org/docs/master/notes/ddp.html#example
    # Avoid the slowing of for loops due to the interpreters GIL.
    # Will spin up independent interpreters, rather than multithreading,
    # as in `DataParallel` case
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # loss_function = WeightedBCELoss(per_channel=True)
    loss_function = WeightedBCEDiceLoss(per_channel=True)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=20
    )

    trainer = RunTraining(
        model,
        device,
        data_loader,
        loss_function,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
    )

    # Run training/validation
    trainer.fit()


if __name__ == "__main__":
    main()
