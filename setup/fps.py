import time
from threading import Lock
from setup.constants import *

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

fps_mutex = Lock()
