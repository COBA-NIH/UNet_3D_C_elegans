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

neptune_run = {}

parser = argparse.ArgumentParser(description="3DUnet Training")

# nargs="?" required to fall back to default if no arg provided
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
    "patch_size": (24, 200, 200),
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

neptune_run["parameters"] = params

train_transforms = [
    aug.Normalize(**params["Normalize"]),
    aug.RandomContrastBrightness(**params["RandomContrastBrightness"]),
    aug.Flip(**params["Flip"]),
    aug.RandomRot90(**params["RandomRot90"]),
    aug.RandomGuassianBlur(**params["RandomGuassianBlur"]),
    aug.RandomGaussianNoise(**params["RandomGaussianNoise"]),
    aug.RandomPoissonNoise(**params["RandomPoissonNoise"]),
    aug.ElasticDeform(**params["ElasticDeform"]),
    aug.LabelsToEdges(**params["LabelsToEdges"]),
    aug.EdgeMaskWmap(**params["EdgeMaskWmap"]),
    # aug.BlurMasks(**params["BlurMasks"]),
    aug.ToTensor()
]
val_transforms = [
    aug.Normalize(**params["Normalize"]),
    aug.LabelsToEdges(**params["LabelsToEdges"]),
    aug.EdgeMaskWmap(**params["EdgeMaskWmap"]),
    # aug.BlurMasks(**params["BlurMasks"]),
    aug.ToTensor()
]

def main():
    main_worker(args)

def main_worker(args):
    if args.dummy:
        print("----- Using dummy data ------")
        train_ds = RandomData(
            data_shape=(1, 1, *params["patch_size"]),
            dataset_size=20,
            num_classes=1,
            train_val="train"
        )
        val_ds = RandomData(
            data_shape=(1, 1, *params["patch_size"]), 
            dataset_size=5, 
            num_classes=1, 
            train_val="val"
        )
    else:
        load_csv = pd.read_csv(args.data)
        # Create the dataset (patches and weight maps, if required)
        utils.create_patch_dataset(load_csv, patch_size=params["patch_size"], create_wmap=params["create_wmap"])
        training_data = pd.read_csv("training_data.csv")
        train_dataset, val_dataset = sklearn.model_selection.train_test_split(
            training_data, test_size=params["val_split"]
        )
        print(
            f"loading data from: {args.data}. Train data of length {train_dataset.shape[0]} and val data of length {val_dataset.shape[0]}"
        )
        train_ds = CElegansDataset(data_csv=train_dataset, transforms=train_transforms, targets=params["targets"], train_val="train")

        val_ds = CElegansDataset(data_csv=val_dataset, transforms=val_transforms, targets=params["targets"], train_val="val")

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
        num_workers=args.workers,
    )

    # Don't shuffle validation so you can see how predictions improve over time
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        num_workers=args.workers,
    )

    data_loader = {"train": train_loader, "val": val_loader}

    model = UNet3D(
        in_channels=params["in_channels"], out_channels=1, f_maps=32
    )

    # model = utils.load_weights(
    #     model, 
    #     weights_path="../3DUnet_confocal_boundary-best_checkpoint.pytorch", 
    #     device="cpu", # Load to CPU and convert to GPU later
    #     dict_key="model_state_dict"
    # )


    model = utils.load_weights(
        model, 
        weights_path="weights/best_checkpoint_exp_044.pytorch", 
        device="cpu", # Load to CPU and convert to GPU later
        dict_key="state_dict"
    )

    model = utils.set_parameter_requires_grad(model, trainable=True)

    model.encoders[0].basic_module.SingleConv1 = SingleConv(params["in_channels"], 16)

    # Replace final sigmoid
    model.final_activation = nn.Identity()

    params_to_update = utils.find_parameter_requires_grad(model)

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

    loss_function = params["loss_function"]()

    # optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(params_to_update, lr=params["lr"], weight_decay=params["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=params["scheduler_mode"], factor=params["scheduler_factor"], patience=params["scheduler_patience"]
    )

    trainer = RunTraining(
        model,
        device,
        data_loader,
        loss_function,
        optimizer,
        scheduler,
        num_epochs=params["epochs"],
        neptune_run=None
    )

    # Run training/validation
    trainer.fit()

    if args.withinference:
        # Run inference pipeline

        load_data_train_no_lab = pd.read_csv("data/data_test_stacked_channels.csv")
        load_data_test = pd.read_csv("data/data_stacked_channels_training.csv")
        load_data_test = load_data_test[load_data_test["train"] == False]

        load_data = pd.concat([load_data_train_no_lab, load_data_test])
        load_data.reset_index(inplace=True, drop=True)

        model = UNet3D(
            in_channels=params["in_channels"], out_channels=params["out_channels"], f_maps=32
        )

        try:
            model = utils.load_weights(
                model, 
                weights_path="best_checkpoint.pytorch", 
                device="cpu", # Load to CPU and convert to GPU later
                dict_key="state_dict"
            )
        except:
            model = utils.load_weights(
                model, 
                weights_path="../best_checkpoint.pytorch", 
                device="cpu", # Load to CPU and convert to GPU later
                dict_key="state_dict"
            )

        model.to("cuda")

        infer = Inferer(
            model=model, 
            patch_size=params["patch_size"],
            neptune_run=None
            )

        infer.predict_from_csv(load_data)

    neptune_run.stop()

if __name__ == "__main__":
    main()
