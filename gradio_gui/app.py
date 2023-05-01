import gradio
from unet.networks.unet3d import UNet3D
from unet.utils.inferer import Inferer
import unet.utils.data_utils as utils
import torch
import pandas as pd

inputs = [
    gradio.Files(file_count="multiple", label="Drag and drop all images to be analyzed. Expected (z, x, y) dimension order 3D tiff files"),
    gradio.Textbox(value="24, 100, 100", interactive=True, label="Patch size (z, x, y)"),
    gradio.Number(value=0.75, interactive=True, label="Patch overlap"),
    gradio.Radio(["gaussian","constant"], type="value", value="gaussian", label="Patch overlap mode"),
    gradio.Number(value=1, interactive=True, label="Batch size for inference"),
    gradio.Number(value=15, interactive=True, label="Object minimum size"),
    gradio.Number(value=250, interactive=True, label="Object maximum size"),
    gradio.Number(value=0.5, interactive=True, label="Threshold"),
]

outputs = [
    gradio.File(label="Download output")
]

def run_inference(image_path_list, patch_size, patch_overlap, patch_mode, batch_size, min_size, max_size, threshold):

    # Convert the patch string into a tuple (not ideal...)
    patch_size = tuple([int(i) for i in patch_size.split(', ')])
    # Batch size needs to be an int
    batch_size = int(batch_size)

    # Create inference DataFrame for a single image
    inference_data_csv = pd.DataFrame({
        "images":[image_path.name for image_path in image_path_list],
        "masks": [None] * len(image_path_list),
        "train": [False] * len(image_path_list)
    })

    # Instantiate model
    model = UNet3D(
        in_channels=2, out_channels=1, f_maps=32
    )

    # Load weights
    model = utils.load_weights(
        model, 
        weights_path="/workspace/best_checkpoint.pytorch", 
        device="cpu", # Load to CPU and convert to GPU later
        dict_key="state_dict"
    )

    if torch.cuda.is_available():
        # Find fastest conv
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print("---- Model running on GPU ----")
    else:
        print("---- Model running on CPU ----")
        device = torch.device("cpu")

    model.to(device)

    # Instantiate the inferer
    infer = Inferer(
        model=model, 
        patch_size=patch_size,
        batch_size=batch_size,
        overlap=patch_overlap,
        patch_mode=patch_mode,
        min_size=min_size,
        max_size=max_size,
        threshold=threshold
        )
    
    # If all paths contain the nd2 extension, load nd2 files. 
    # Assumes heterogeenous files are not uploaded
    img_format = all([i for i in image_path_list if ".nd2" in i])

    # Run inference on csv 
    infer.predict_from_csv(inference_data_csv, from_nd2=img_format)

    output_paths = inference_data_csv.loc[:, "segmentation"].tolist()

    print("Finished!")

    return output_paths

app = gradio.Interface(
    fn=run_inference, 
    inputs=inputs, 
    outputs=outputs,
    title="3D UNet C. elegans"
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)  