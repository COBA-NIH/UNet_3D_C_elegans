import pandas as pd
from unet.networks.unet3d import UNet3D
from unet.utils.inferer import Inferer
import unet.utils.data_utils as utils
import torch
import numpy as np

load_data_train_no_lab = pd.read_csv("data/data_test_stacked_channels.csv")
load_data_test = pd.read_csv("data/data_stacked_channels_training.csv")
load_data_test = load_data_test[load_data_test["train"] == False]

load_data = pd.concat([load_data_train_no_lab, load_data_test])
load_data.reset_index(inplace=True, drop=True)

model = UNet3D(
    in_channels=2, out_channels=1, f_maps=32
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
    patch_size=[24, 400, 400]
    )

infer.predict_from_csv(load_data)
