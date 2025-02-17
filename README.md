Clone this [repo](https://github.com/marcoslucianops/DeepStream-Yolo.git) and export your yolov8 model utils/export_yoloV8.py. Once done copy the generated onnx file inside the DeepStream-Yolo folder then execute 
```bash
export CUDA_VER=12.6 
```
and then execute this 

```bash
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```
Start docker by executing 
```bash
source docker.sh
```

