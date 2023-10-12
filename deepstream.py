import sys
import time
import numpy as np
import cv2
import gi
import pyds
from gi.repository import GObject, Gst, GLib
from threading import Lock

# Constants
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
TENSOR_SHAPE = 128
CONFIG_PATH = "config_infer_primary_yoloV8.txt"  # Ensure this path is correct

# PGIE Classes
PGIE_CLASSES = {
    'male_less_than_27': 0,
    'female_less_than_27': 1,
    'male_27_to_48': 2,
    'female_27_to_48': 3,
    'male_greater_than_48': 4,
    'female_greater_than_48': 5
}
pgie_classes_str = list(PGIE_CLASSES.keys())

classification_results = []
face_counter = []

# OSD Settings
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1
MUXER_BATCH_TIMEOUT_USEC = 4000000

# Detector UIDs
PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2

# Secondary Detector Class IDs
SGIE_CLASS_ID_FACE = 0
SGIE_CLASS_ID_LP = 1

# TODO: Move sensitive information like RTSP sources and credentials to a configuration file or environment variables
SOURCES = [
    "rtsp://USERNAME:PASSWORD@IP_ADDRESS:PORT",
    "rtsp://USERNAME:PASSWORD@IP_ADDRESS:PORT",
    "rtsp://USERNAME:PASSWORD@IP_ADDRESS:PORT",
    "rtsp://USERNAME:PASSWORD@IP_ADDRESS:PORT"
]
NUM_SOURCES = len(SOURCES)

fps_mutex = Lock()


class GETFPS:
    def __init__(self, stream_id):
        self.start_time = time.time()
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        if end_time - self.start_time > 5:
            print("**********************FPS*****************************************")
            print("Fps of stream", self.stream_id, "is ", float(self.frame_count) / 5.0)
            self.frame_count = 0
            self.start_time = end_time
        else:
            self.frame_count += 1

    def print_data(self):
        print('frame_count=', self.frame_count)
        print('start_time=', self.start_time)


class PERF_DATA:
    def __init__(self, num_streams=1):
        self.all_stream_fps = {"stream{}".format(i): GETFPS(i) for i in range(num_streams)}

    def perf_print_callback(self):
        self.perf_dict = {idx: stream.get_fps() for idx, stream in self.all_stream_fps.items()}
        print("\n**PERF:", self.perf_dict, "\n")
        return True

    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_fps()

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top, left, width, height = int(rect_params.top), int(rect_params.left), int(rect_params.width), int(rect_params.height)

    color = (255, 0, 0, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)

    def draw_corner_line(image, start, end, color):
        """Draws a line on the image from start to end with the given color."""
        return cv2.line(image, start, end, color, 2)

    # Draw top and bottom lines
    image = draw_corner_line(image, (left + w_percents, top), (left + width - w_percents, top), color)
    image = draw_corner_line(image, (left + w_percents, top + height), (left + width - w_percents, top + height), color)
    # Draw left and right lines
    image = draw_corner_line(image, (left, top + h_percents), (left, top + height - h_percents), color)
    image = draw_corner_line(image, (left + width, top + h_percents), (left + width, top + height - h_percents), color)

    return image

faces_embeddings = []  # Replace with your actual embeddings list
labels = []  # Replace with your actual labels list

def predict_using_classifier(embeddings, labels, face_embedding, data_to_send):
    # Placeholder function. Replace with your actual classifier's prediction method.
    return "predicted_label", 0.95, "user_id"

def sgie_sink_pad_buffer_probe(pad, info, u_data):
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    # Extracting bounding boxes and class scores from YOLO's output
    bounding_boxes, class_scores = extract_yolo_output(gst_buffer)

    # Process the bounding boxes and class scores as required
    # This could involve drawing the bounding boxes, filtering based on confidence thresholds, etc.

    for box, score in zip(bounding_boxes, class_scores):
        # Example: Drawing the bounding box
        draw_bounding_boxes(image, box, score)
        
    return Gst.PadProbeReturn.OK

def extract_yolo_output(buffer):
    # Placeholder function to demonstrate extracting bounding boxes and class scores from YOLO's output
    # The exact extraction process will depend on the YOLO output format

    bounding_boxes = []
    class_scores = []

    # Assuming buffer contains the raw tensor output from YOLOv8
    tensor_output = np.frombuffer(buffer, dtype=np.float32)

    # Reshape the tensor_output based on YOLO's expected output shape. 
    # This is just a placeholder and may need to be adjusted based on the specific YOLOv8 configuration.
    tensor_output = tensor_output.reshape(-1, 85)  # Assuming 80 classes + 4 for bbox coordinates + 1 for objectness score

    for item in tensor_output:
        objectness_score = item[4]
        class_probabilities = item[5:]
        bbox = item[:4]

        # Find the class with the highest probability
        class_index = np.argmax(class_probabilities)
        class_score = class_probabilities[class_index]

        # Filtering based on a confidence threshold, for example 0.5
        if objectness_score * class_score > 0.5:
            bounding_boxes.append(bbox)
            class_scores.append((class_index, class_score))

    return bounding_boxes, class_scores


def osd_sink_pad_buffer_probe(pad, info, u_data):
    vehicle_count, person_count, face_count, lp_count, num_rects = 0, 0, 0, 0, 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    class_str = pgie_classes_str[obj_meta.class_id]
                    if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                        if class_str in ['male_less_than_27', 'male_27_to_48', 'male_greater_than_48']:
                            person_count += 1
                        elif class_str in ['female_less_than_27', 'female_27_to_48', 'female_greater_than_48']:
                            person_count += 1
                    if obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                        if obj_meta.class_id == SGIE_CLASS_ID_FACE: face_count += 1
                        if obj_meta.class_id == SGIE_CLASS_ID_LP: lp_count += 1
                    l_obj = l_obj.next
                except StopIteration: break
            face_counter.append(face_count)
            print(f"Read frame {frame_number} from {frame_meta.source_id}")
            l_frame = l_frame.next
        except StopIteration:
            print("Errored at FD")
            break

    return Gst.PadProbeReturn.OK



def integrated_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()

    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Extract bounding box and save image
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
            frame_copy = np.array(n_frame, copy=True, order='C')
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

            # Handle object metadata and analytics
            l_user_meta = obj_meta.obj_user_meta_list
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    # Handle analytics extraction here if needed

                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break

            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK



def cb_newpad(decodebin, decoder_src_pad, data):
    # Log the entry to the function for debugging
    print("In cb_newpad")

    # Get the capabilities of the decoder source pad
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    features = caps.get_features(0)

    # Debug prints to understand the flow and values
    print("Pad capabilities name:", gstname)
    print("Pad capabilities features:", features)

    # Check if the pad is for video and not audio
    if "video" in gstname:
        # Ensure the decodebin has picked the NVIDIA decoder plugin
        if features.contains("memory:NVMM"):
            # Link the decodebin pad to the source bin ghost pad
            bin_ghost_pad = data.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick the NVIDIA decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.set_property("download", True)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True



def main(args):
    # Initialization
    Gst.init(None)
    pipeline = Gst.Pipeline()

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Failed to create StreamMux\n")
        return -1
    pipeline.add(streammux)

    # Source Bins & Pad Linking
    for i in range(NUM_SOURCES):
        uri_name = SOURCES[i]
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write(f"Failed to create source bin for {uri_name}\n")
            return -1
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write(f"Failed to get request pad {padname} from {uri_name}\n")
            return -1
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # Queue Elements
    queue_elements = [Gst.ElementFactory.make("queue", f"queue{i}") for i in range(1, 6)]
    for queue in queue_elements:
        pipeline.add(queue)

    # Additional Gstreamer Elements
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Failed to create primary-inference\n")
        return -1
    pgie.set_property('config-file-path', "config_infer_primary_yoloV8.txt")

    face_detector = Gst.ElementFactory.make("nvinfer", "face-detector")
    if not face_detector:
        sys.stderr.write("Failed to create face-detector\n")
        return -1

    face_classifier = Gst.ElementFactory.make("nvinfer", "face-classifier")
    if not face_classifier:
        sys.stderr.write("Failed to create face-classifier\n")
        return -1

    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write("Failed to create analytics\n")
        return -1
    pipeline.add(pgie, face_detector, face_classifier, nvanalytics)

    # Pipeline Construction
    streammux.link(queue_elements[0])
    queue_elements[0].link(face_detector)
    face_detector.link(queue_elements[1])
    queue_elements[1].link(face_classifier)
    face_classifier.link(queue_elements[2])
    queue_elements[2].link(nvanalytics)
    
    # Example continuation for linking pipeline elements
    another_element = Gst.ElementFactory.make("some_element", "element_name")
    pipeline.add(another_element)
    queue_elements[3].link(another_element)

    yet_another_element = Gst.ElementFactory.make("another_element", "another_element_name")
    pipeline.add(yet_another_element)
    another_element.link(yet_another_element)
    yet_another_element.link(queue_elements[4])

    # Attach the probes to the appropriate elements
    sgie_src_pad = face_classifier.get_static_pad("src")
    if not sgie_src_pad:
        sys.stderr.write("Unable to get src pad from face_classifier\n")
    else:
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_sink_pad_buffer_probe, 0)

    osd_src_pad = nvanalytics.get_static_pad("src")
    if not osd_src_pad:
        sys.stderr.write("Unable to get src pad from nvanalytics\n")
    else:
        osd_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    nvanalytics_src_pad = nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write("Unable to get src pad\n")
        return -1
    else:
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, integrated_sink_pad_buffer_probe, 0)

    # Playback
    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    try:
        loop.run()
    except Exception as e:
        print(f"Error: {e}")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))