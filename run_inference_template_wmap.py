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
#from unet.utils.dummy_model import DummyModel

output_folder = "output" 

neptune_run = neptune.init_run(
    tags=["testing_neptune_on"],
    project="BroadImagingPlatform/maddox",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MDliZmIxMS02NjNhLTQ0OTMtYjYwMS1lOWM3N2ZmMjdlYzAifQ==",
    custom_run_id="Maddox_id_test_6", # Usamos el mismo run_id que en el código A
)


#class DummyModel(torch.nn.Module):
 #   def forward(self, x):
  #      return x


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

#model = DummyModel()
model.to("cuda")

infer = Inferer(
    model=model, 
    patch_size=[24, 100, 100]
    )

infer.predict_from_csv(load_data)

for filename in os.listdir(output_folder):
    if filename.endswith(".tiff"):
        # Full path to the TIFF file
        file_path = os.path.join(output_folder, filename)

        # Read the TIFF stack
        tiff_stack = tifffile.imread(file_path)

        # Select the desired plane, z10
        plane = tiff_stack[10]
        # Normalizar y convertir a uint16
        plane_16bit = np.uint16(65535 * (plane - np.min(plane)) / (np.max(plane) - np.min(plane)))

        # Guardar la imagen como TIFF (formato uint16)
        tiff_filename = f"{os.path.splitext(filename)[0]}_plane10.tiff"


#        plane_16bit = np.uint16(255 * (plane - np.min(plane)) / (np.max(plane) - np.min(plane)))  # Normalizar a [0, 255]
#        img = Image.fromarray(plane_16bit)
        neptune_run["Prediction_images_z10"].append(plane_16bit)
#        neptune_run["Prediction"].append(File.as_image(plane))

output_tar_gz = "output.tar.gz"
folder_to_compress = "output"
with tarfile.open(output_tar_gz, "w:gz") as tar:
    tar.add(folder_to_compress, arcname=os.path.basename(folder_to_compress))  # Compress folder

neptune_run["Prediction/images_tif"].upload("output.tar.gz")
