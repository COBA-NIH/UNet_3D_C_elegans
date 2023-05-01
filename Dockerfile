# docker build -t ctromanscoia/unet3d_c_elegans:0.1 .
# docker run -it --rm -p 127.0.0.1:8000:8000 ctromanscoia/unet3d_c_elegans:0.1

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Install multicut and its deps
RUN conda install -c conda-forge nifty python-elf

# Install other UNet deps
RUN python -m pip install gradio==3.25.0 scikit-image numpy monai pandas neptune-client==0.16.18 nd2

ENV HOME=/workspace

## Add files to the container
# Add weights
ADD weights/best_checkpoint_exp_044.pytorch /workspace/best_checkpoint.pytorch
# Add the UNet files
ADD unet/ /workspace/unet
# Add gradio app
ADD gradio_gui/app.py /workspace/app.py

# Run the gradio app
CMD [ "python3" , "/workspace/app.py" ]
