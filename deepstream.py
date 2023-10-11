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
import time
from threading import Lock

import platform
import sys

import gi
import sys
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst






def is_aarch64():
    return platform.uname()[4] == 'aarch64'

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')


start_time=time.time()

fps_mutex = Lock()

class GETFPS:
    def __init__(self,stream_id):
        global start_time
        self.start_time=start_time
        self.is_first=True
        self.frame_count=0
        self.stream_id=stream_id

    def update_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        else:
            global fps_mutex
            with fps_mutex:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        end_time = time.time()
        with fps_mutex:
            stream_fps = float(self.frame_count/(end_time - self.start_time))
            self.frame_count = 0
        self.start_time = end_time
        return round(stream_fps, 2)

    def print_data(self):
        print('frame_count=',self.frame_count)
        print('start_time=',self.start_time)

class PERF_DATA:
    def __init__(self, num_streams=1):
        self.perf_dict = {}
        self.all_stream_fps = {}
        for i in range(num_streams):
            self.all_stream_fps["stream{0}".format(i)]=GETFPS(i)

    def perf_print_callback(self):
        self.perf_dict = {stream_index:stream.get_fps() for (stream_index, stream) in self.all_stream_fps.items()}
        print ("\n**PERF: ", self.perf_dict, "\n")
        return True
    
    def update_fps(self, stream_index):
        self.all_stream_fps[stream_index].update_fps()

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





def draw_bounding_boxes(image, obj_meta, confidence):

    rect_params = obj_meta.rect_params
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




if __name__ == '__main__':
    sys.exit(main(sys.argv))
