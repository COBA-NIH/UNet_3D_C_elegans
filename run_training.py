import sys
sys.path.append("../")
import os
import shutil
import tempfile
import time
import numpy as np
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
import logging

from torch.utils.data import DataLoader, Dataset

import torch
torch.cuda.empty_cache()
import torchvision
import torch.nn as nn
import skimage
import pandas as pd
import sklearn
import sklearn.model_selection

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
from unet.utils.load_data import MaddoxDataset
from unet.networks.threedee_unet import UNet_3D

load_csv = pd.read_csv("patch_data/load_data_training.csv")

train_dataset, val_dataset = sklearn.model_selection.train_test_split(load_csv, test_size=0.2)

train_ds = MaddoxDataset(
    data_csv=train_dataset,
    train_val="train"
)

val_ds = MaddoxDataset(
    data_csv=val_dataset,
    train_val="val"
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=train_ds.collate_fn)

# Don't shuffle validation so you can see how predictions improve over time
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=val_ds.collate_fn)

def train_step(
    model,
    device,
    data_loader,
    loss_fn,
    optimizer
):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        # Put the image and the mask on the device
        X, y = X.to(device), y.to(device)
        # Clear gradients from the optimizer
        optimizer.zero_grad()
        # Run a prediction
        prediction = model(X)
        print(f"Prediction shape: {prediction.shape} --- Target shape: {y.shape}")
        # Find the loss of the prediction when compared to the GT
        loss = loss_fn(prediction, y)
        # Using this loss, calculate the gradients of loss for all parameters
        loss.backward()
        # Update the the parameters of the model
        optimizer.step()

        # Sigmoid scales values between 0-1
        probability = prediction.sigmoid().detach().cpu().numpy().flatten()
        y = y.long().detach().cpu().numpy().flatten()

        # roc = sklearn.metrics.roc_auc_score(y, probability, labels=[0, 1], multi_class="ovo")

        # Assume that correct predictions are above threshold
        # Use correctly classified pixels to calculate precision, recall, accuracy and f1
        th_prediction = (probability > 0.5)
        precision = sklearn.metrics.precision_score(y, th_prediction, labels=[0, 1])
        recall = sklearn.metrics.recall_score(y, th_prediction, labels=[0, 1])
        accuracy = sklearn.metrics.accuracy_score(y, th_prediction)
        f1 = sklearn.metrics.f1_score(y, th_prediction, labels=[0, 1])

        logger.log(
            level=logging.INFO, 
            msg=f"Train step: [{i} of {len(data_loader)}] Loss: {loss.item()}, Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}"
            )

        # print(
        #     f"""
        #     Training step: {i} of {len(data_loader)}
        #     Loss: {loss}
        #     Accuracy: {accuracy}
        #     Precision: {precision}
        #     Recall: {recall}
        #     F1: {f1}
        #     """
        # )

def validation_step(
    model,
    device,
    data_loader,
    loss_fn
):
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            print(f"Prediction shape: {prediction.shape} --- Target shape: {y.shape}")
            loss = loss_fn(prediction, y)

            probability = prediction.sigmoid().detach().cpu().numpy().flatten()
            y = y.long().cpu().detach().numpy().flatten()

            # roc = sklearn.metrics.roc_auc_score(y, probability, labels=[0, 1], multi_class="ovo")
            
            th_prediction = (probability > 0.5)
            precision = sklearn.metrics.precision_score(y, th_prediction, labels=[0, 1])
            recall = sklearn.metrics.recall_score(y, th_prediction, labels=[0, 1])
            accuracy = sklearn.metrics.accuracy_score(y, th_prediction)
            f1 = sklearn.metrics.f1_score(y, th_prediction, labels=[0, 1])


            logger.log(
                level=logging.INFO, 
                msg=f"Valid step: [{i} of {len(data_loader)}] Loss: {loss.item()}, Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}"
                )

            # print(
            #     f"""
            #     Validation step: {i} of {len(data_loader)}
            #     Loss: {loss}
            #     Accuracy: {accuracy}
            #     Precision: {precision}
            #     Recall: {recall}
            #     F1: {f1}
            #     """
            # )

max_epochs = 2

device = "cuda"
model = UNet_3D(
    in_channels=1, 
    out_channels=2
    ).to(device)

logger = logging.getLogger('training')
logger = logging.getLogger("training")
log_outfile = logging.FileHandler("experiment.log")
log_outfile.setLevel(logging.DEBUG)
logger.addHandler(log_outfile)

if torch.cuda.device_count() > 1:
    print("Running on multiple GPUs")
    model = nn.DataParallel(model)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=False)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

for epoch in range(max_epochs):
    print("epoch:", epoch)
    train_step(model, device, train_loader, loss_function, optimizer)
    validation_step(model, device, val_loader, loss_function)



