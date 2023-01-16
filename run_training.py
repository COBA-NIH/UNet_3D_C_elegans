import sys
sys.path.append("../")
import os
import shutil
import tempfile
import time
import numpy as np
# from monai.losses import DiceLoss
# from monai.inferers import sliding_window_inference
import logging
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# http://localhost:6006

import torch
import torchvision
import torch.nn as nn
import skimage
import pandas as pd
import sklearn
import sklearn.model_selection

from unet.utils.load_data import MaddoxDataset, RandomData
from unet.networks.unet3d import UNet3D
from unet.utils.loss import BCEDiceLoss
from unet.utils.trainer import RunTraining

import argparse


parser = argparse.ArgumentParser(description="3DUnet Training")

# nargs="?" required to fall back to default if no arg provided
parser.add_argument("data", nargs="?",  default="patch_data/load_data_training.csv")

def main():
    args = parser.parse_args()

    main_worker(args)

def main_worker(args):
    print(f"loading data from: args.data")
    load_csv = pd.read_csv(args.data)

    train_dataset, val_dataset = sklearn.model_selection.train_test_split(load_csv, test_size=0.2)

    # train_ds = MaddoxDataset(
    #     data_csv=train_dataset,
    #     train_val="train"
    # )

    # val_ds = MaddoxDataset(
    #     data_csv=val_dataset,
    #     train_val="val"
    # )


    train_ds = RandomData(
        data_shape=(1, 4, 32, 32), 
        dataset_size=20,
        num_classes=2,
        train_val="train"
    )

    val_ds = RandomData(
        data_shape=(1, 4, 32, 32), 
        dataset_size=5,
        num_classes=2,
        train_val="val"
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=train_ds.collate_fn)

    # Don't shuffle validation so you can see how predictions improve over time
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)

    data_loader = {
        "train": train_loader,
        "val": val_loader
    }

    device = "cpu"
    model = UNet3D(
        input_channels=1, 
        num_classes=2,
        network_depth=2,
        activation="sigmoid"
        ).to(device)

    if torch.cuda.device_count() > 1:
        print("Running on multiple GPUs")
        model = nn.DataParallel(model)


    loss_function = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20)

    trainer = RunTraining(
        model,
        device,
        data_loader,
        loss_function,
        optimizer,
        scheduler,
        num_epochs=10
    )

    # Run training/validation
    trainer.fit()
        

if __name__ == "__main__":
    main()



