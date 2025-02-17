Clone this [repo](https://github.com/marcoslucianops/DeepStream-Yolo.git) and export your yolov8 modle using utils/export_yoloV8.py. Once done copy the generated onnx file inside the DeepStream-Yolo folder then execute 

export CUDA_VER=12.6 ## please check the version once
and then execute this 

make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

Start docker by executing docker.sh

