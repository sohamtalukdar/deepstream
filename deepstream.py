
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
# from common.FPS import PERF_DATA
import uuid
import datetime
import pyds
import cv2
import numpy as np
import base64
import threading,queue
import os


# perf_data = None
MAX_DISPLAY_LEN=64 
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



# nvanlytics_src_pad_buffer_probe  will extract metadata received on nvtiler sink pad
# and update params for drawing rectangle, object information etc.
def nvanalytics_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0   
    num_rects=0
    gst_buffer = info.get_buffer()
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    

    while l_frame:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        # obj_counter = {
        # PGIE_CLASS_ID_male_less_than_27:0,
        # PGIE_CLASS_ID_female_less_than_27:1,
        # PGIE_CLASS_ID_male_27_to_48:2,
        # PGIE_CLASS_ID_female_27_to_48:3,
        # PGIE_CLASS_ID_male_greater_than_48:4,
        # PGIE_CLASS_ID_female_greater_than_48:5,
    
        
        # }
       
        # print("#"*50)
        while l_obj:
            try: 
                # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                # The casting is done by pyds.NvDsObjectMeta.cast()
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1
            l_user_meta = obj_meta.obj_user_meta_list
            # points = np.array(points, dtype=np.int32)
            
           

            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
            # n_frame = pyds.get_nvds_buf_surface()
            frame_copy = np.array(n_frame, copy=True, order='C')
            #print(np.shape(frame_copy))
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
            rect_params = obj_meta.rect_params
            x, y, width, height = rect_params.top, rect_params.left, rect_params.width, rect_params.height   
            x, y, width, height = int(x), int(y), int(width), int(height)
            # x = max(0, x)
            # y = max(0, y)
            # width = min(width, frame_copy.shape[1] - x)
            # height = min(height, frame_copy.shape[0] - y)  
            cropped_image = frame_copy[x:x+height,y:y+width]  

            # results, boxes = CharDetection(cropped_image)
      
            
            
            img_path ="1.jpg"
            cv2.imwrite(img_path, cropped_image)
            SourceID = frame_meta.source_id
            SourceID =str(SourceID)


            

            # Extract object level meta data from NvDsAnalyticsObjInfo
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                    # if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):             
                    #     user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                    #     if user_meta_data.dirStatus: print("Object {0} moving in direction: {1}".format(obj_meta.object_id, user_meta_data.dirStatus))                    
                    #     if user_meta_data.lcStatus: print("Object {0} line crossing status: {1}".format(obj_meta.object_id, user_meta_data.lcStatus))
                    #     if user_meta_data.ocStatus: print("Object {0} overcrowding status: {1}".format(obj_meta.object_id, user_meta_data.ocStatus))
                    #     if user_meta_data.roiStatus: print("Object {0} roi status: {1}".format(obj_meta.object_id, user_meta_data.roiStatus))
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
    
        # Get meta data from NvDsAnalyticsFrameMeta
        l_user = frame_meta.frame_user_meta_list
        print("Frame Number=", frame_number, "stream id=", frame_meta.pad_index, "Number of Objects=",num_rects,)
       
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
                    # if user_meta_data.objInROIcnt: print("Total Person in ROI: {0}".format(user_meta_data.objInROIcnt))
                    PerconCountInROI = user_meta_data.objInROIcnt
              
                    if user_meta_data.ocStatus: print("Overcrowding status: {0}".format(user_meta_data.ocStatus))
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break
        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        # global perf_data
        # perf_data.update_fps(stream_index)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        # pritesting video/6.mp4nt("#"*50)

    return Gst.PadProbeReturn.OK

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    # obj_name = pgie_classes_str[obj_meta.class_id]
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
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
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    # image = cv2.putText(image, obj_name, (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     (255, 0, 0, 0), 1)
    return image

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

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
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

def is_aarch64():
    return platform.uname()[4] == 'aarch64'

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

def main(args):
    # filename = os.path.basename(__file__)
    # print(filename[:-3])
    # config = configparser.ConfigParser()

    # config.read('config.ini')

    # variables = config.items(str(filename[:-3]))

    # cam_dict = {}

    # for variable in variables:
    #     cam_dict[variable[0]] = variable[1]

    # values = list(cam_dict.values())
    # num_cams = len(values)

    # if num_cams == 1:
    #     cam1 = values[0]
    #     sources = [cam1]
    # elif num_cams == 2:
    #     cam1, cam2 = values
    #     sources = [cam1,cam2]
    # elif num_cams == 3:
    #     cam1, cam2, cam3 = values
    #     sources = [cam1,cam2,cam3]
    # elif num_cams ==4:
    #     cam1,cam2,cam3,cam4 = values
    #     sources = [cam1,cam2,cam3,cam4]


    # sources = ["file:///home/dev1/Projects/mall_age_gender/testing_video/6.mp4"]
    sources = [
        "rtsp://ranu:Bharat1947@192.168.177.36:554",
        "rtsp://ranu:Bharat1947@192.168.177.36:554",
        "rtsp://ranu:Bharat1947@192.168.177.36:554",
        "rtsp://ranu:Bharat1947@192.168.177.36:554",
        ]
    number_sources = len(sources)
    # global perf_data
    # perf_data = PERF_DATA(len(args) - 1)
    # number_sources=len(args)-1

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
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
    
    # tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    nvanalytics.set_property("config-file", "deepstream_app_config.txt")
    
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
        




    pgie.set_property('config-file-path', "config_infer_primary_yoloV8.txt")
    # pgie_batch_size=pgie.get_property("batch-size")
    # if(pgie_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
    #     pgie.set_property("batch-size",number_sources)

    # tiler_rows=int(math.sqrt(number_sources))
    # tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    # tiler.set_property("rows",tiler_rows)
    # tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    # #Set properties of tracker
    # config = configparser.ConfigParser()
    # config.read('infer_tracker_config.txt')
    # config.sections()

    # for key in config['tracker']:
    #     if key == 'tracker-width' :
    #         tracker_width = config.getint('tracker', key)
    #         tracker.set_property('tracker-width', tracker_width)
    #     if key == 'tracker-height' :
    #         tracker_height = config.getint('tracker', key)
    #         tracker.set_property('tracker-height', tracker_height)
    #     if key == 'gpu-id' :
    #         tracker_gpu_id = config.getint('tracker', key)
    #         tracker.set_property('gpu_id', tracker_gpu_id)
    #     if key == 'll-lib-file' :
    #         tracker_ll_lib_file = config.get('tracker', key)
    #         tracker.set_property('ll-lib-file', tracker_ll_lib_file)
    #     if key == 'll-config-file' :
    #         tracker_ll_config_file = config.get('tracker', key)
    #         tracker.set_property('ll-config-file', tracker_ll_config_file)
    #     if key == 'enable-batch-process' :
    #         tracker_enable_batch_process = config.getint('tracker', key)
    #         tracker.set_property('enable_batch_process', tracker_enable_batch_process)
    #     if key == 'enable-past-frame' :
    #         tracker_enable_past_frame = config.getint('tracker', key)
    #         tracker.set_property('enable_past_frame', tracker_enable_past_frame)

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

    # streammux.link(pgie)
    # pgie.link(nvconv1)
    # nvconv1.link(filter1)
    # filter1.link(nvanalytics)
    # nvanalytics.link(nvosd)
    # nvosd.link(sink)
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
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)
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

if __name__ == '__main__':
    sys.exit(main(sys.argv))