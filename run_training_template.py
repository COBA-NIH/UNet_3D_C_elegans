import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
import sklearn.model_selection
from unet.utils.load_data import MaddoxDataset, RandomData
from unet.networks.unet3d import UNet3D
from unet.networks.unet3d import SingleConv
# from unet.networks.unet3d import UnetModel
import unet.augmentations.augmentations as aug
from unet.utils.loss import WeightedBCELoss, WeightedBCEDiceLoss, BCEDiceLoss, BCEDiceLossAlt
from unet.utils.trainer import RunTraining
import argparse
import unet.utils.data_utils as utils


parser = argparse.ArgumentParser(description="3DUnet Training")

# nargs="?" required to fall back to default if no arg provided
parser.add_argument("data", nargs="?")
parser.add_argument("--batch", nargs="?", default=4, type=int)
parser.add_argument("--epochs", nargs="?", default=10, type=int)
parser.add_argument("--workers", nargs="?", default=4, type=int)
parser.add_argument("--dummy", action="store_true")  # Use dummy data

train_transforms = [
    aug.RandomContrastBrightness(p=0.5),
    aug.Flip(p=0.5),
    aug.RandomRot90(p=0.5, channel_axis=0),
    aug.RandomGuassianBlur(p=0.5),
    aug.RandomGaussianNoise(p=0.5),
    aug.RandomPoissonNoise(p=0.5),
    aug.ElasticDeform(sigma=10, p=0.5, channel_axis=0, mode="mirror"),
    aug.LabelsToEdges(connectivity=2),
    aug.EdgeMaskWmap(),
    # aug.BlurMasks(sigma=2),
    aug.Normalize(),
    aug.ToTensor()
]
val_transforms = [
    aug.LabelsToEdges(connectivity=2),
    aug.EdgeMaskWmap(),
    # aug.BlurMasks(sigma=2),
    aug.Normalize(),
    aug.ToTensor()
]

targets=[["image"], ["mask"], ["weight_map"]]

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
        # Create the dataset (patches and weight maps, if required)
        utils.create_patch_dataset(load_csv, (24, 200, 200), create_wmap=True, w0=3)
        training_data = pd.read_csv("training_data.csv")
        train_dataset, val_dataset = sklearn.model_selection.train_test_split(
            training_data, test_size=0.2
        )
        print(
            f"loading data from: {args.data}. Train data of length {train_dataset.shape[0]} and val data of length {val_dataset.shape[0]}"
        )
        train_ds = MaddoxDataset(data_csv=train_dataset, transforms=train_transforms, targets=targets, train_val="train", wmap=True)

        val_ds = MaddoxDataset(data_csv=val_dataset, transforms=val_transforms, targets=targets, train_val="val", wmap=True)

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
        in_channels=1, out_channels=1, f_maps=32
    )

    # model = utils.load_weights(
    #     model, 
    #     weights_path="../3DUnet_confocal_boundary-best_checkpoint.pytorch", 
    #     device="cpu", # Load to CPU and convert to GPU later
    #     dict_key="model_state_dict"
    # )

    model = utils.set_parameter_requires_grad(model, trainable=True)

    model.encoders[0].basic_module.SingleConv1 = SingleConv(2, 16)

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

    loss_function = WeightedBCEDiceLoss()

    # optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(params_to_update, 1e-4, weight_decay=1e-5)

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
