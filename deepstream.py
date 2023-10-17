import sys
import time
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0') 
from gi.repository import Gst, GLib
import pyds
from threading import Lock
import argparse
import os

# Constants
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
fps_mutex = Lock()

class GETFPS:
    """Class to measure FPS for streams."""
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

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    color = (255, 0, 0, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 2)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 2)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 2)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 2)
    return image

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
            dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            draw_bounding_boxes(obj_meta, dummy_image)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        stream_key = 'stream{}'.format(current_index)
        if stream_key in fps_streams:
            fps_streams[stream_key].update_and_get_fps()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)
    if gstname.find("video") != -1 and features.contains("memory:NVMM"):
        bin_ghost_pad = source_bin.get_static_pad("src")
        if not bin_ghost_pad.set_target(decoder_src_pad):
            sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

def create_source_bin(index, uri):
    print("Creating source bin")
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def cb_probe(pad, info, user_data):
    return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stderr.write("Received EOS. Exiting ...\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

if __name__ == '__main__':
    Gst.init(None)
    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline()
    for i in range(NUM_SOURCES):
        uri_name = SOURCES[i]
        fps_streams["stream{0}".format(i)] = GETFPS(i)
        source = create_source_bin(i, uri_name)
        pipeline.add(source)
        bin_pad = source.get_static_pad("src")
        if not bin_pad:
            sys.stderr.write("Failed to get src pad of source bin \n")
        sink_pad = streammux.get_request_pad("sink_%u")
        if not sink_pad:
            sys.stderr.write("Streammux request sink pad failed. Exiting. \n")
            sys.exit(1)
        if not bin_pad.link(sink_pad):
            sys.stderr.write("Failed to link source bin to stream muxer \n")
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)
