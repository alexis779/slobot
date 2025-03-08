import time
import json

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.fps_metric import FpsMetric


class FpsGauge():
    def __init__(self, **kwargs):
        self.max_period = kwargs['max_period']
        self.res = kwargs['res']

    def show_fps(self):
        self.previous_time = time.time()
        self.frames = 0

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, frame_handler=self, res=self.res, show_viewer=False)
        arm.elemental_rotations()

        arm.genesis.stop()

    def handle_frame(self, frame):
        self.frames += 1

        current_time = time.time()

        diff_time = current_time - self.previous_time
        if diff_time > self.max_period:
            fps = self.frames / diff_time
            self.add_metric(current_time, fps)
            self.frames = 0
            self.previous_time = current_time

    def add_metric(self, timestamp, fps):
        fps_metric = FpsMetric(timestamp, fps)
        print("FPS=", fps_metric)