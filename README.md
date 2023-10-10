# deepstream

## Setup Virtual Environment (venv):

```
pip3 install virtualenvwrapper
mkvirtualenv deepstream
```

## Install required packages:
```
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```
## Export YOLOv8 Model:

```
python3 export_yoloV8.py -w <your_weights> --dynamic
```
## Compile YOLO Custom Layers for DeepStream:

```
CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo
```
Note: Configuration Adjustments:

Update YOLO Configuration: Modify the config_infer_primary_yolov8.txt file to reflect the correct number of classes for your YOLO model.

Update DeepStream Configuration: Change the URL in deepstream_app_config to point to your data source.

## Run DeepStream Application:
```
deepstream-app -c deepstream_app_config.txt
```
