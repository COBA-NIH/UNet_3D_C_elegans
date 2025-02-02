import pandas as pd
import os
from unet.networks.unet3d import UNet3D
from unet.utils.inferer import Inferer
import unet.utils.data_utils as utils
import torch
import numpy as np
import neptune.new as neptune

# Cargar el run_id desde el archivo
#with open("run_id.txt", "r") as f:
 #   run_id = f.read().strip()

# Inicializa el run de Neptune usando el run_id
neptune_run = neptune.init_run(
    tags=["testing_neptune_on"],
    project="BroadImagingPlatform/maddox",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MDliZmIxMS02NjNhLTQ0OTMtYjYwMS1lOWM3N2ZmMjdlYzAifQ==",
    custom_run_id="Maddox_id_test", # Usamos el mismo run_id que en el código A
)
# Obtener el directorio actual y el directorio padre
current_dir_inf = os.getcwd()
parent_dir_inf = os.path.abspath(os.path.join(current_dir_inf, os.pardir))

# Listar archivos en ambos directorios
current_dir_files_inf = os.listdir(current_dir_inf)
parent_dir_files_inf = os.listdir(parent_dir_inf)

# Registrar información en Neptune
neptune_run["directory_inf/current_path_inference"].log(current_dir_inf)
neptune_run["directory_inf/parent_path_inference"].log(parent_dir_inf)
neptune_run["directory_inf/current_files_inference"].log(current_dir_files_inf)
neptune_run["directory_inf/parent_files_inference"].log(parent_dir_files_inf)


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
    patch_size=[24, 150, 150],
    neptune_run=neptune_run
    )

infer.predict_from_csv(load_data)

#updated_data = infer.predict_from_csv(load_data)

#Upload images to neptune

#for i, row in updated_data.iterrows():
 #   if not pd.isna(row["prediction"]):
  #      prediction = skimage.io.imread(row["prediction"]).astype(np.float32)
   #     norm_prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    #    neptune_run[f"predictions/{os.path.basename(row['prediction'])}"].upload(
     #       File.as_image((norm_prediction * 255).astype(np.uint8))
      #  )
#    if not pd.isna(row["segmentation"]):
 #       segmentation = skimage.io.imread(row["segmentation"])
  #      neptune_run[f"segmentations/{os.path.basename(row['segmentation'])}"].upload(
   #         File.as_image(segmentation)
    #    )
