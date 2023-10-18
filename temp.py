
from asyncore import file_dispatcher
import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import math
import platform
import uuid
import datetime
import pyds
import cv2
import numpy as np
import base64
import threading,queue
import os

PGIE_CLASS_ID_male_less_than_27 = 0
PGIE_CLASS_ID_female_less_than_27 = 1
PGIE_CLASS_ID_male_27_to_48 = 2
PGIE_CLASS_ID_female_27_to_48 = 3
PGIE_CLASS_ID_male_greater_than_48 =4
PGIE_CLASS_ID_female_greater_than_48 = 5

MUXER_OUTPUT_WIDTH=1980
MUXER_OUTPUT_HEIGHT=1020
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1980
TILED_OUTPUT_HEIGHT=1020
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

pgie_classes_str= ['male_less_than_27', 'female_less_than_27', 'male_27_to_48','female_27_to_48','male_greater_than_48', 'female_greater_than_48']

# File paths
input_file_list = [
    "file:///home/soham/deepstream/video.mp4"

]
output_file = "output.mp4" # Output file location
pgie_config_file = "config_infer_primary_yoloV8.txt"  # Path to pgie config file
nvanalytics_file = "deepstream_app_config.txt"  # Path to nvanalytics config file
# Tracker options
#enable_tracker = 1 # Enable/disable tracker and SGIEs. 0 for disable, 1 for enable

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

def nvanalytics_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0   
    num_rects=0
    gst_buffer = info.get_buffer()    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
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

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        while l_obj:
            try: 
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            l_user_meta = obj_meta.obj_user_meta_list
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
            frame_copy = np.array(n_frame, copy=True, order='C')
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
            rect_params = obj_meta.rect_params
            x, y, width, height = rect_params.top, rect_params.left, rect_params.width, rect_params.height   
            x, y, width, height = int(x), int(y), int(width), int(height)
            cropped_image = frame_copy[x:x+height,y:y+width]  
            img_path ="1.jpg"
            cv2.imwrite(img_path, cropped_image)
            SourceID = frame_meta.source_id
            SourceID =str(SourceID)
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                except StopIteration:
                    break
                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        l_user = frame_meta.frame_user_meta_list
        print("Frame Number=", frame_number, "stream id=", frame_meta.pad_index, "Number of Objects=",num_rects,)
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
                    PerconCountInROI = user_meta_data.objInROIcnt
                    if user_meta_data.ocStatus: print("Overcrowding status: {0}".format(user_meta_data.ocStatus))
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break
        stream_index = "stream{0}".format(frame_meta.pad_index)
        try:
            l_frame=l_frame.next
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
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
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
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
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
    sources = ["file:///home/soham/deepstream/video.mp4"]
    # sources = [
    #     "rtsp://ranu:Bharat1947@192.168.177.36:554"
    #     ]
    number_sources = len(sources)
    Gst.init(None)
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False
    print("Creating streamux \n ")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=sources[i]
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
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 400000)
    streammux.set_property('attach-sys-ts', True)
    streammux.set_property('compute-hw',1)
    streammux.set_property('live-source',1)
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    nvanalytics.set_property("config-file", nvanalytics_file)
    
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    # nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    # nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    # sink = Gst.ElementFactory.make("fakesink", "fakesink")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    # sink = Gst.ElementFactory.make("filesink","filesink")
    # sink.set_property("location","./out.mp4")
    sink.set_property('sync',0)
    sink.set_property('qos',0)
        
    pgie.set_property('config-file-path', pgie_config_file)

    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
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

# ## ###### #####################################################################
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
    queue9.link(tiler)
    nvosd.link(sink)
    queue10.link(tiler)
    nvosd.link(sink)
    queue11.link(tiler)
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
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)

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

if __name__ == '__main__':
    sys.exit(main(sys.argv))