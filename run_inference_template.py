import pandas as pd
from unet.networks.unet3d import UNet3D
from unet.utils.inferer import Inferer
import unet.utils.data_utils as utils
import torch
import numpy as np
import neptune.new as neptune
#from neptune.types import File
import tarfile
import tifffile
from PIL import Image
import os

output_folder = "output" 

neptune_run = neptune.init_run(
    tags=["testing_neptune_on"],
    project="BroadImagingPlatform/maddox",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MDliZmIxMS02NjNhLTQ0OTMtYjYwMS1lOWM3N2ZmMjdlYzAifQ==",
    custom_run_id="gt3_repeat_185" # Usamos el mismo run_id que en el código A
)

load_data_train_no_lab = pd.read_csv("data/data_test_stacked_channels.csv")
load_data_test = pd.read_csv("data/data_stacked_channels_training.csv")
load_data_test = load_data_test[load_data_test["train"] == False]

neptune_run["datasets/test_data"].upload("data/data_test_stacked_channels.csv")
neptune_run["datasets/train_data"].upload("data/data_stacked_channels_training.csv")

load_data = pd.concat([load_data_train_no_lab, load_data_test])
load_data.reset_index(inplace=True, drop=True)

model = UNet3D(
    in_channels=2, out_channels=2, f_maps=32
)

try:
    model = utils.load_weights(
        model, 
        weights_path="maddox_190.pytorch", 
        device="cpu", # Load to CPU and convert to GPU later
        dict_key="state_dict"
    )
except:
    model = utils.load_weights(
        model, 
        weights_path="../maddox_190.pytorch", 
        device="cpu", # Load to CPU and convert to GPU later
        dict_key="state_dict"
    )

model.to("cuda")

infer = Inferer(
    model=model, 
    patch_size=[24, 150, 150]
    )

infer.predict_from_csv(load_data)

output_tar_gz = "output.tar.gz"
folder_to_compress = "output"
with tarfile.open(output_tar_gz, "w:gz") as tar:
    tar.add(folder_to_compress, arcname=os.path.basename(folder_to_compress))  # Compress folder

neptune_run["Prediction/images_tif"].upload("output.tar.gz")
