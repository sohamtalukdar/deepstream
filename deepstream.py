from sqlite3 import Time, Timestamp
import sys

sys.path.append('../')
sys.path.append('/opt/nvidia/deepstream/deepstream-6.0/lib')
import gi
import configparser
import json
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import cv2
import os
import os.path
from os import path
# from s3_upload import upload_to_s3
from facenet_utils import load_dataset, normalize_vectors, predict_using_classifier
import datetime 
import ctypes
import pyds
import config as cf
from PIL import Image

fps_stream=None
face_counter= []
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_PERSON = 2

SGIE_CLASS_ID_LP = 1
SGIE_CLASS_ID_FACE = 0
GST_CAPS_FEATURES_NVMM="memory:NVMM"
pgie_classes = ["Vehicle", "TwoWheeler", "Person", "Roadsign"]

PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2
DATASET_PATH = './encodings_facenet/test.npz'
#DATASET_PATH = 'bsnl-faces-embeddings.npz'

faces_embeddings, labels = load_dataset(DATASET_PATH)
data = {"encodings": faces_embeddings, "names": labels}
# labels1 = labels[0]
# labels2 = labels[4]
# labels = [labels1 ,labels2]
#print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk",data)
fps_streams = {}
frame_count = {}
saved_count = {}


def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    global save_count
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    # frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

    # frame = pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
    # frame_copy = np.array(frame,copy=True,order='C')
    # frame_copy = cv2.cvtColor(frame_copy,cv2.COLOR_RGBA2BGRA)
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            name = pyds.get_string(obj_meta.text_params.display_text)
            values = name.split(" ")
            #Values = "{face} {EMPID} {Probab} {UUID}"
            # print("from tiler "+ str(obj_meta.confidence))
            #(values)
            if len(values) < 2:
                try:
                    l_obj = l_obj.next
                    continue
                except StopIteration:
                    break
            #print("this is values",values)
            if values[1] not in ["Unknown"]:
                #print(values[1])
                # print("A")
                if True:
                    # print("B")
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    # the input should be address of buffer and batch_id
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    # print("WPEF")
                    n_frame = draw_bounding_boxes(n_frame, obj_meta, 0.53)
                    # convert python array into numpy array format in the copy mode.
                    frame_copy = np.array(n_frame, copy=True, order='C')
                    # print(np.shape(n_frame))
                    # convert the array into cv2 default color format
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    # print("lllllllllllllllllll",pyds.get_string(obj_meta.text_params.display_text))
                    x = datetime.datetime.now()
                    d = x.strftime('%x')
                    d = d.split("/")
                    d = str("{}-{}-{}".format(d[1],d[0],d[2]))
                    t = x.strftime("%X")
                    nt = t.split(":")
                    minute = int(nt[1])
                    print(minute)

                    for m in range(minute,minute+1):    
                        if not os.path.exists("./cropped_images/{}/{}:{}".format(d,nt[0],nt[1])):                    
                            dir = os.makedirs("./cropped_images/{}/{}:{}".format(d,nt[0],nt[1]))
                        img_path = "cropped_images/{}/{}:{}/{} at {}.jpeg".format(d,nt[0],nt[1],values[1],t)
                        cv2.imwrite(img_path, frame_copy)
                        print(img_path)

                
                    

                    

                    # img_path = "cropped_images/{}.jpeg".format(values[1])
                    # # print(img_path)
                    # cv2.imwrite(img_path, frame_copy)


                save_image = True

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

            # if save_image:
            #     img_path = "1.jpeg"
            #     print(img_path)
            #     cv2.imwrite(img_path, frame_copy)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def draw_bounding_boxes(image, obj_meta, confidence):

    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    # obj_name = 'face'
    obj_name = pyds.get_string(obj_meta.text_params.display_text).split(' ')
    image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2)
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    # image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height) 
    linebot_c2 = (left + width - w_percents, top + height)
    # image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    # image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    # image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # print(f"From bbox {obj_name}")
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name[1] + " " + obj_name[2], (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255, 0), 2)
    return image

def sgie_sink_pad_buffer_probe(pad,info,u_data):
    
    frame_number=0
    
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
   
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # print(dir(frame_meta))
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        #obj_meta = frame_meta.obj_meta

        #print(obj_meta.rect_params)

        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            l_user = obj_meta.obj_user_meta_list
            top = obj_meta.rect_params.top
            left = obj_meta.rect_params.left

            while l_user is not None:
                
                try:
                    # Casting l_user.data to pyds.NvDsUserMeta
                    user_meta=pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    continue
                
                # Converting to tensor metadata
                # Casting user_meta.user_meta_data to NvDsInferTensorMeta
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                
                # Get output layer as NvDsInferLayerInfo 
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                # Convert NvDsInferLayerInfo buffer to numpy array
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                v = np.ctypeslib.as_array(ptr, shape=(128,))
                # print(v)
                
                # Pridict face neme
                yhat = v.reshape((1,-1))
                
                face_to_predict_embedding = normalize_vectors(yhat)
                # SourceID = str(frame_meta.source_id)
                # GateInfo = cf.GateID[SourceID]
                SourceId = frame_meta.source_id
                SourceId = str(SourceId)
                GateInfo = cf.GateID[SourceId]
                # print(frame_meta.frame_num)
                #print(GateStatus,GateInfo[0],GateInfo[1])
                dict = {'CameraID':GateInfo[1],'CameraName':GateInfo[0]}
                DataToSend = '{%s}' % ', '.join(['"%s": "%s"' % (k, v) for k, v in dict.items()])
                result,conf, uid = predict_using_classifier(faces_embeddings, labels, face_to_predict_embedding,DataToSend)
                # result =  (str(result).title())
                # print('Predicted name: %s' % result
                # write_meta(gst_buffer, res=result)
                

                # Generate classifer metadata and attach to obj_meta

                
                # Get NvDsClassifierMeta object 
                classifier_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)

                # Pobulate classifier_meta data with pridction result
                classifier_meta.unique_component_id = tensor_meta.unique_id
                
                
                label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)

                
                label_info.result_prob = 0
                label_info.result_class_id = 0

                pyds.nvds_add_label_info_meta_to_classifier(classifier_meta, label_info)
                pyds.nvds_add_classifier_meta_to_object(obj_meta, classifier_meta)
                
                # pyds.nvds_add_user_meta_to_obj(obj_meta, m)

                display_text = pyds.get_string(obj_meta.text_params.display_text)
                # print(f'From sgie {display_text}')
                obj_meta.text_params.display_text = f'{display_text} {result} {conf} {uid}'

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
        except StopIteration:
            print("Errored at FC")
            break
    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
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

def osd_sink_pad_buffer_probe(pad,info,u_data):
    global fps_stream, face_counter
    frame_number=0
    #Intiallizing object counter with 0.
    vehicle_count = 0
    person_count = 0
    face_count = 0
    lp_count = 0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                if obj_meta.class_id == PGIE_CLASS_ID_VEHICLE:
                   vehicle_count += 1
                if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                   person_count += 1

            if obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                if obj_meta.class_id == SGIE_CLASS_ID_FACE:
                   face_count += 1
                if obj_meta.class_id == SGIE_CLASS_ID_LP:
                   lp_count += 1
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

    # print(obj_meta.rect_params)
    #fps_stream.get_fps()
    # Acquiring a display meta object. The memory ownership remains in
    # the C code so downstream plugins can still access it. Otherwise
    # the garbage collector will claim it when this probe function exits.
    #display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    #display_meta.num_labels = 1
    #py_nvosd_text_params = display_meta.text_params[0]
    # Setting display text to be shown on screen
    # Note that the pyds module allocates a buffer for the string, and the
    # memory will not be claimed by the garbage collector.
    # Reading the display_text field here will return the C address of the
    # allocated string. Use pyds.get_string() to get the string content.
    # py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={}  Person_count={} Face Count={}".format(frame_number, num_rects, person_count, face_count)
        face_counter.append(face_count)
        print(f"Read frame {frame_number} from {frame_meta.source_id}")

        # Now set the offsets where the string should appear
        # py_nvosd_text_params.x_offset = 10
        #py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        #py_nvosd_text_params.font_params.font_name = "Serif"
        #py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        #py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        #py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        #py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        #pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            print("Errored at FD")
            break
                        
    return Gst.PadProbeReturn.OK	

def main(args):
    # Check input arguments


    
    # sources = ['rtsp://10.192.13.30:554/Live/0/Main','rtsp://10.192.13.32:554/Live/0/Main',
    # 'rtsp://10.192.13.33:554/Live/0/Main','rtsp://10.192.13.34:554/Live/0/Main',
    # #'rtsp://10.192.13.35:554/Live/0/Main',
    # 'rtsp://10.192.13.36:554/Live/0/Main',]
    #"rtsp://ranu:Bharat1947@192.168.1.66:554/Profile2/media.smp"
    sources = [cf.Cam1]
    

    number_sources = len(sources)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    face_detector = Gst.ElementFactory.make("nvinfer", "primary-inference")
    face_classifier = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    
    # streammux.set_property('config-file-path', "streammux_config.txt")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', len(sources))
    #streammux.set_property('batched-push-timeout', 4000000)
    streammux.set_property('batched-push-timeout', 400000)
    streammux.set_property('attach-sys-ts', True)
    streammux.set_property('compute-hw',1)
    streammux.set_property('live-source',1)
    # streammux.set_property('sync-inputs',0)
    face_detector.set_property('config-file-path', "./configs/detector_config.txt")
    face_classifier.set_property('config-file-path', "./configs/classifier_config.txt")

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")

    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    # nvconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    filter1.set_property("caps", caps1)
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    sink.set_property('sync',0)
    sink.set_property('qos',0)

    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(face_classifier)
    pipeline.add(face_detector)
    pipeline.add(nvconv1)
    pipeline.add(filter1)
    # pipeline.add(tiler)
    pipeline.add(nvosd)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        # nvconv.set_property("nvbuf-memory-type", mem_type)
        nvconv1.set_property("nvbuf-memory-type", mem_type)
 

    
    pipeline.add(sink)
    lat = pipeline.get_latency()


    streammux.link(queue1)
    queue1.link(face_detector)
    face_detector.link(queue2)
    queue2.link(face_classifier)
    face_classifier.link(queue3)
    queue3.link(nvconv1)
    nvconv1.link(filter1)
    # filter1.link(tiler)
    filter1.link(queue4)
    queue4.link(nvosd)
    # nvconv.link(nvosd)
    # tiler.link(nvosd)
    nvosd.link(sink)
    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # streammux_sinkpad = queue1.get_static_pad("sink")

    # streammux_sinkpad.add_probe(Gst.PadProbeType.BUFFER, get_frame_num,0)

    osdsinkpad = queue2.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    vidconvsinkpad = queue3.get_static_pad("sink")
    if not vidconvsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvvidconv \n")

    vidconvsinkpad.add_probe(Gst.PadProbeType.BUFFER, sgie_sink_pad_buffer_probe, 0)

    save_sinkpad = nvosd.get_static_pad("sink")
    save_sinkpad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
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
