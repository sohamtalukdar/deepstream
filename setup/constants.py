# Constants and configuration values
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
PGIE_CLASSES = {
    'male_less_than_27': 0, 'female_less_than_27': 1, 'male_27_to_48': 2, 
    'female_27_to_48': 3, 'male_greater_than_48': 4, 'female_greater_than_48': 5
}
pgie_classes_str = list(PGIE_CLASSES.keys())
classification_results = []
face_counter = []
fps_streams = {}
SOURCE = ''
CONFIG_INFER = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1984
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5
SOURCES = ["file:///home/soham/deepstream/video.mp4"]
NUM_SOURCES = len(SOURCES)
