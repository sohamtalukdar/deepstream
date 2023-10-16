import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    # Initialize TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create a network with the kEXPLICIT_BATCH flag set
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load the ONNX model and parse it to populate the TensorRT network
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Create an optimization profile and set its dimensions
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    min_shape = (1, 3, 640, 640)
    opt_shape = (1, 3, 640, 640)
    max_shape = (1, 3, 640, 640)

    profile.set_shape(network.get_input(0).name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    config.max_workspace_size = 1 << 20
    engine = builder.build_engine(network, config)

    # Save the engine to a file
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    return engine

# Convert the ONNX model to a TensorRT engine
build_engine("yolov8l.onnx", "yolov8l.engine")
