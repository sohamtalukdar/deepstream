import sys
import gi
gi.require_version('Gst', '1.0') 
from gi.repository import Gst, GLib

from setup.constants import *
from setup.utils import *
from setup.fps import *

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