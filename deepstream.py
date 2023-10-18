import sys
import time
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0') 
from gi.repository import GObject, Gst, GLib
import pyds
from threading import Lock
import platform
from ctypes import *
import ctypes
import os
import argparse
import platform


# Constants
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
TENSOR_SHAPE = 128

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

fps_streams = {}

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ''
CONFIG_INFER = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5

# OSD Settings
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1
MUXER_BATCH_TIMEOUT_USEC = 4000000

# Detector UIDs
PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2


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
        self.total_fps_time = 0
        self.total_frame_count = 0

    def update_and_get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            self.total_fps_time += current_time
            self.total_frame_count += self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            print(f'DEBUG: FPS of stream {self.stream_id + 1}: {current_fps:.2f} ({avg_fps:.2f})')
            self.start_time = end_time
            self.frame_count = 0
        else:
            self.frame_count += 1
        return current_fps, avg_fps

class PERF_DATA:
    def __init__(self, num_streams=1):
        self.all_stream_fps = {f"stream{i}": GETFPS(i) for i in range(num_streams)}

    def perf_print_callback(self):
        self.perf_dict = {idx: stream.update_and_get_fps() for idx, stream in self.all_stream_fps.items()}
        print("\n**PERF:", self.perf_dict, "\n")
        return True

    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_and_get_fps()

def draw_bounding_boxes(obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    # For displaying age and gender
    obj_name = pyds.get_string(obj_meta.text_params.display_text).split(' ')

    # Drawing the bounding box rectangle (from second snippet)
    image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2)

    # Drawing corner lines (from the first snippet)
    color = (255, 0, 0, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    
    # Top and bottom lines
    image = cv2.line(image, (left + w_percents, top), (left + width - w_percents, top), color, 2)
    image = cv2.line(image, (left + w_percents, top + height), (left + width - w_percents, top + height), color, 2)
    
    # Left and right lines
    image = cv2.line(image, (left, top + h_percents), (left, top + height - h_percents), color, 2)
    image = cv2.line(image, (left + width, top + h_percents), (left + width, top + height - h_percents), color, 2)

    # Displaying age and gender on the bounding box
    image = cv2.putText(image, obj_name[1] + " " + obj_name[2], (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 0), 2)

    return image

def parse_face_from_meta(frame_meta, obj_meta):
    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH,
               obj_meta.mask_params.height / STREAMMUX_HEIGHT)
    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for i in range(num_joints):
        data = obj_meta.mask_params.get_mask_array()
        xc = int((data[i * 3 + 0] - pad_x) / gain)
        yc = int((data[i * 3 + 1] - pad_y) / gain)
        confidence = data[i * 3 + 2]

        if confidence < 0.5:
            continue

        if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle_params = display_meta.circle_params[display_meta.num_circles]
        circle_params.xc = xc
        circle_params.yc = yc
        circle_params.radius = 6
        circle_params.circle_color.red = 1.0
        circle_params.circle_color.green = 1.0
        circle_params.circle_color.blue = 1.0
        circle_params.circle_color.alpha = 1.0
        circle_params.has_bg_color = 1
        circle_params.bg_color.red = 0.0
        circle_params.bg_color.green = 0.0
        circle_params.bg_color.blue = 1.0
        circle_params.bg_color.alpha = 1.0
        display_meta.num_circles += 1

def tracker_src_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        current_index = frame_meta.source_id

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            parse_face_from_meta(frame_meta, obj_meta)
            draw_bounding_boxes(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        fps_streams['stream{0}'.format(current_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write('DEBUG: EOS\n')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == 'aarch64'

def main(args):

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create nvstreammux\n")
        return None
    pipeline.add(streammux)

    for i in range(NUM_SOURCES):
        print("Creating source_bin ",i," \n ")
        uri_name=SOURCES[i]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    queue9=Gst.ElementFactory.make("queue","queue9")
    queue10=Gst.ElementFactory.make("queue","queue10")
    queue11=Gst.ElementFactory.make("queue","queue11")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    pipeline.add(queue9)
    pipeline.add(queue10)
    pipeline.add(queue11)
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    
    streammux.set_property('width', 1980)
    streammux.set_property('height', 1020)
    streammux.set_property('batch-size', NUM_SOURCES)
    streammux.set_property('batched-push-timeout', 400000)
    streammux.set_property('attach-sys-ts', True)
    streammux.set_property('compute-hw',1)
    streammux.set_property('live-source',1)
    
    # tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    nvanalytics.set_property("config-file", "deepstream_app_config.txt")
    
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync',0)
    sink.set_property('qos',0)
        
    pgie.set_property('config-file-path', "config_infer_primary_yoloV8.txt")

    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvconv1)
    pipeline.add(filter1)
    # pipeline.add(tracker)
    pipeline.add(nvanalytics)
    pipeline.add(tiler)
    # pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(nvconv1)
    nvconv1.link(filter1)
    filter1.link(queue3)
    queue3.link(nvanalytics)
    nvanalytics.link(queue4)
    queue4.link(tiler)
    tiler.link(queue5)
    queue5.link(nvosd)
    nvosd.link(sink)

    # We link elements in the following order:
    # sourcebin -> streammux -> nvinfer -> nvtracker -> nvdsanalytics ->
    # nvtiler -> nvvideoconvert -> nvdsosd -> sink
    print("Linking elements in the Pipeline \n")

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    nvanalytics_src_pad=nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        # GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)       

def parse_args():
    global SOURCE, CONFIG_INFER, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    parser.add_argument('-s', '--source', required=True, help='Source stream/file')
    parser.add_argument('-c', '--config-infer', required=True, help='Config infer file')
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=1920, help='Streammux width (default: 1920)')
    parser.add_argument('-e', '--streammux-height', type=int, default=1080, help='Streammux height (default: 1080)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '':
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)
    if args.config_infer == '' or not os.path.isfile(args.config_infer):
        sys.stderr.write('ERROR: Config infer not found\n')
        sys.exit(1)

    SOURCE = args.source
    CONFIG_INFER = args.config_infer
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval 

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv[1:]))
