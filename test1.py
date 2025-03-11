import sys
import os
import cv2
import gi
import ctypes
import math
import argparse
import cupy as cp
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

perf_data = None

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

# ---------------------------------------------------------------------------
# The tiler sink pad probe function is called for every frame.
# In this modified version, we also crop out each detected bounding box
# region from the GPU frame and save it as an image in the "output" folder.
# ---------------------------------------------------------------------------
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    # Retrieve batch metadata from the gst_buffer.
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Cast the data to frame metadata.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num

        # Collect all object metadata (bounding boxes) for this frame.
        obj_meta_list = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                obj_meta_list.append(obj_meta)
            except StopIteration:
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Create a dummy owner object for lifetime management.
        owner = None

        # Retrieve the image data from the GPU buffer.
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
            hash(gst_buffer), frame_meta.batch_id)
        # Retrieve the pointer from the PyCapsule.
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        n_frame_gpu = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')

        # (Optional) Modify the frame. For example, add a blue tint to the red channel.
        # stream = cp.cuda.stream.Stream(null=True)
        # with stream:
        #     n_frame_gpu[:, :, 0] = 0.5 * n_frame_gpu[:, :, 0] + 0.5
        # stream.synchronize()

        # For each detected object, crop the bounding box region and save as an image.
        for idx, obj_meta in enumerate(obj_meta_list):
            # Get bounding box coordinates (and cast to int).
            x = int(obj_meta.rect_params.left)
            y = int(obj_meta.rect_params.top)
            w = int(obj_meta.rect_params.width)
            h = int(obj_meta.rect_params.height)

            # Ensure coordinates are within frame bounds.
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > shape[1]:
                w = shape[1] - x
            if y + h > shape[0]:
                h = shape[0] - y
            if w <= 0 or h <= 0:
                continue

            # Crop the region from the GPU frame (results in a cupy array).
            cropped_gpu = n_frame_gpu[y:y+h, x:x+w, :]
            # Convert the cropped region to a numpy array.
            cropped_np = cp.asnumpy(cropped_gpu)
            # Convert from RGBA to BGR (cv2 expects BGR images for saving/viewing).
            cropped_bgr = cv2.cvtColor(cropped_np, cv2.COLOR_RGBA2BGR)

            # Create output directory if it doesn't exist.
            output_folder = "output"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # Build a filename using the frame number and object index.
            filename = os.path.join(output_folder, f"frame_{frame_number}_obj_{idx}.jpg")
            cv2.imwrite(filename, cropped_bgr)
            print(f"Saved bounding box image: {filename}")

        print("Frame Number =", frame_number, "Number of Objects =", len(obj_meta_list))
        stream_index = f"stream{frame_meta.pad_index}"
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------------------------
# Callback for when a new pad is added to the decodebin.
# ---------------------------------------------------------------------------
def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if (gstname.find("video") != -1):
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")

# ---------------------------------------------------------------------------
# Callback for decodebin's child-added signal.
# ---------------------------------------------------------------------------
def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name)
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') is not None:
            Object.set_property("drop-on-latency", True)

# ---------------------------------------------------------------------------
# Create a source bin to wrap a uri decodebin.
# ---------------------------------------------------------------------------
def create_source_bin(index, uri):
    print("Creating source bin")
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None
    return nbin

# ---------------------------------------------------------------------------
# Main pipeline creation and execution.
# ---------------------------------------------------------------------------
def main(args):
    global perf_data
    perf_data = PERF_DATA(len(args))
    number_sources = len(args)

    Gst.init(None)
    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
    print("Creating streammux")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin", i)
        uri_name = args[i]
        if uri_name.startswith("rtsp://"):
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin\n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin\n")
        srcpad.link(sinkpad)
    print("Creating Pgie")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")
    print("Creating nvvidconv1")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write("Unable to create nvvidconv1\n")
    print("Creating filter1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write("Unable to get the caps filter1\n")
    filter1.set_property("caps", caps1)
    print("Creating tiler")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler\n")
    print("Creating nvvidconv")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")
    print("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")

    if PlatformInfo().is_platform_aarch64():
        print("Creating nv3dsink")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    else:
        print("Creating EGLSink")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write("Unable to create egl sink\n")

    if is_live:
        print("At least one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', "test1.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, "with number of sources", number_sources)
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    print("Adding elements to Pipeline")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add the pad probe to the tiler's sink pad.
    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write("Unable to get tiler sink pad\n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ":", source)

    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        print("Error:", e)
    print("Exiting app")
    pipeline.set_state(Gst.State.NULL)

# ---------------------------------------------------------------------------
# Parse command-line arguments.xlear
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="deepstream_imagedata-multistream_cupy.py",
        description="This app takes multiple URI streams as input, retrieves the image "
                    "buffer from GPU as a cupy array for in-place modification, and extracts "
                    "bounding boxes as images saved to a folder."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        required=True,
    )
    args = parser.parse_args()
    stream_paths = args.input
    return stream_paths

if __name__ == '__main__':
    platform_info = PlatformInfo()
    if platform_info.is_integrated_gpu():
        sys.stderr.write("\nThis app is not currently supported on integrated GPU. Exiting...\n\n")
        sys.exit(1)
    stream_paths = parse_args()
    sys.exit(main(stream_paths))
