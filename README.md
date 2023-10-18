# deepstream

## Install required packages

```shell
pip3 install -r requirements.txt
python3 setup.py install
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_x86_64.whl
pip3 install pyds-1.1.8-py3-none-linux_x86_64.whl
```

## Export YOLOv8 Model

```shell
python3 export_yoloV8.py -w <your_weights> --dynamic
```

Note: Change the name of onnx file accordingly in engine.py

## Compile YOLO Custom Layers for DeepStream

```shell
export CUDA_VER=11.8   #make sure to check the version of your cuda
make -C nvdsinfer_custom_impl_Yolo
```

Note: Configuration Adjustments:

Update YOLO Configuration: Modify the config_infer_primary_yolov8.txt file to reflect the correct number of classes for your YOLO model.

Update DeepStream Configuration: Change the URL in deepstream_app_config to point to your data source.

***Run the main file***