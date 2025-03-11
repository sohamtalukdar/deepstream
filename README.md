Clone this [repo](https://github.com/marcoslucianops/DeepStream-Yolo.git) and export your yolov8 models using export_yoloV8.py that can be found under utils. Once done copy the generated onnx file inside the root DeepStream-Yolo folder then execute:

Make sure to install the devel deepstream7.1

Start docker by executing 
```bash
source docker.sh
```

then install the dependencies
```bash
source install_dependencies.sh
```
Exporting the yolo model

```bash
python3 export_yoloV8.py -w your-model.pt
```

```bash
export CUDA_VER=12.6 
```
and then execute this 

```bash
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```
Now make sure to copy the libnvdsinfer_custom_impl_Yolo.so in your project root folder from DeepStream-Yolo/nvdsinfer_custom_impl_Yolo.  Make sure the config files (test1.txt) have these params set:

```bash
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/workspace/libnvdsinfer_custom_impl_Yolo.so
```
Then test using this:
```bash
python3 test1.py -i file:///workspace/your-file
```

**Note**:
Please make sure that the config file test1.txt aligns well with your models. You can use Netron/Nsight DL designer. Make sure atleast the Onnx file & the labels file is present. If once the testing result seems convinving, then make sure to convert the model to engine file only in the same docker env to that of deepstream with the required config(i.e. Int8 or fp16- by default acceptance).