# docker build -t pllanosf/unet3d_c_elegans_app:0.1 .
# docker run -it --rm -p 127.0.0.1:8000:8000 pllanosf/unet3d_c_elegans_app:0.1

FROM ctromanscoia/unet3d_c_elegans:0.4

ENV HOME=/workspace

## Add files to the container
# Add weights
ADD weights/maddox_239.pytorch /workspace/maddox_239.pytorch
# Add the UNet files
ADD unet/ /workspace/unet
# Add gradio app
ADD gradio_gui/app.py /workspace/app.py

# Run the gradio app
CMD [ "python3" , "/workspace/app.py" ]
