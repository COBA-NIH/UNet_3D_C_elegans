import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
import sklearn.model_selection
from unet.utils.load_data import CElegansDataset, RandomData
from unet.networks.unet3d import UNet3D
from unet.networks.unet3d import SingleConv
# from unet.networks.unet3d import UnetModel
import unet.augmentations.augmentations as aug
from unet.utils.loss import WeightedBCELoss, WeightedBCEDiceLoss, BCEDiceLoss
from unet.utils.trainer import RunTraining
from unet.utils.inferer import Inferer
import argparse
import unet.utils.data_utils as utils

import neptune.new as neptune


parser.add_argument("data", nargs="?")
parser.add_argument("--batch", nargs="?", default=4, type=int)
parser.add_argument("--epochs", nargs="?", default=10, type=int)
parser.add_argument("--workers", nargs="?", default=4, type=int)
parser.add_argument("--dummy", action="store_true")  # Use dummy data
parser.add_argument("--withinference", action="store_true")

args = parser.parse_args()

params = {
    "Normalize": {"per_channel": True},
    "RandomContrastBrightness": {"p": 0.5},
    "Flip": {"p": 0.5},
    "RandomRot90": {"p": 0.5, "channel_axis": 0},
    "RandomGuassianBlur": {"p": 0.5},
    "RandomGaussianNoise": {"p": 0.5},
    "RandomPoissonNoise": {"p": 0.5},
    "ElasticDeform": {"sigma":10, "p":0.5, "channel_axis": 0, "mode":"mirror"},
    "LabelsToEdges": {"connectivity": 2, "mode":"thick"},
    "EdgeMaskWmap": {"edge_multiplier":2, "wmap_multiplier":1, "invert_wmap":True},
    # "BlurMasks": {"sigma": 2},
    "ToTensor": {},
    "batch_size": args.batch,
    "epochs": args.epochs,
    "val_split": 0.2,
    "patch_size": (24, 150, 150),
    "create_wmap": True, ##
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "in_channels": 2,
    "out_channels": 1,
    "scheduler_factor": 0.2,
    "scheduler_patience": 20,
    "scheduler_mode": "min",
    "loss_function": WeightedBCEDiceLoss,
    # "loss_function": BCEDiceLoss,
    # "targets": [["image"], ["mask"]]
    "targets": [["image"], ["mask"], ["weight_map"]]
}



def main():
    main_worker(args)

def main_worker(args):
    load_csv = pd.read_csv(data)
        # Create the dataset (patches and weight maps, if required)
    utils.create_patch_dataset(load_csv, patch_size=params["patch_size"], create_wmap=params["create_wmap"])
     #   training_data = pd.read_csv("training_data.csv")
      #  train_dataset, val_dataset = sklearn.model_selection.train_test_split(
      #      training_data, test_size=params["val_split"]
      #  )
#        train_ds = CElegansDataset(data_csv=train_dataset, transforms=train_transforms, targets=params["targets"], train_val="train")

 #       val_ds = CElegansDataset(data_csv=val_dataset, transforms=val_transforms, targets=params["targets"], train_val="val")

  #  if torch.cuda.is_available():
        # Find fastest conv
   #     torch.backends.cudnn.benchmark = True
    #    device = torch.device("cuda")
   # else:
    #    device = torch.device("cpu")

   # train_loader = DataLoader(
    #    train_ds,
     #   batch_size=args.batch,
     #   shuffle=True,
      #  pin_memory=True if device == "cuda" else False,
      #  num_workers=args.workers,
   # )

    # Don't shuffle validation so you can see how predictions improve over time
   # val_loader = DataLoader(
    #    val_ds,
     #   batch_size=args.batch,
      #  shuffle=False,
       # pin_memory=True if device == "cuda" else False,
       # num_workers=args.workers,
   # )


if __name__ == "__main__":
    main()
