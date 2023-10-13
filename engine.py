import tensorrt as trt
import onnx
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
model_path = "/home/soham/deepstream/New_yolo_V8M.onnx"
with open(model_path, "rb") as f:
    onnx_model = f.read()
batch_size = 1
builder.max_batch_size = batch_size
builder.max_workspace_size = 1 << 20  # Specify maximum workspace size
builder.fp16_mode = False  # Use FP32 precision
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
if not parser.parse(onnx_model):
    for error in range(parser.num_errors):
        print(parser.get_error(error))
engine = builder.build_cuda_engine(network)

engine_path = "model_b1_gpu0_fp32.engine"
with open(engine_path, "wb") as f:
    f.write(engine.serialize())

del engine
del network
del parser

