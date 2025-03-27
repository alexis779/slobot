import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import uuid
import os

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.simulation_frame import SimulationFrame

# Generate a stream of images from the simulation
class ImageQueue:

    def __init__(self):
        os.makedirs(Configuration.WORK_DIR, exist_ok=True)

    def simulation_frames(self, max_fps, res):
        self.frames = []

        self.min_period = 1.0 / max_fps
        self.previous_time = time.time()

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, frame_handler=self, res=res, show_viewer=False)
        arm.elemental_rotations()

        # stop genesis
        arm.genesis.stop()

        return self.frames

    def handle_frame(self, frame):
        current_time = time.time()
        diff_time = current_time - self.previous_time
        if diff_time >= self.min_period:
            self.add_frame(frame)
            self.previous_time = current_time

    def add_frame(self, frame):
        rgb = frame[0]

        depth = frame[1]
        depth = self._logarithmic_depth_to_rgb(depth)

        segmentation = frame[2]
        surface = frame[3]

        frame = (rgb, depth, segmentation, surface)

        simulation_frame = self.create_simulation_frame(frame)
        self.frames.append(simulation_frame)

    def create_simulation_frame(self, frame) -> SimulationFrame:
        timestamp = time.time()

        image_paths = list(map(self.create_image_paths, frame))

        return SimulationFrame(timestamp, image_paths)

    def create_image_paths(self, image_array):
        image_name = str(uuid.uuid4())
        image_path = os.path.join(Configuration.WORK_DIR, f"{image_name}.webp")
        image = Image.fromarray(image_array, mode='RGB')
        image.save(image_path)
        return image_path

    def _logarithmic_depth_to_rgb(self, depth_arr):
        """
        Use logarithmic scaling to enhance depth visualization
        Helps spread out colors more non-linearly, potentially improving contrast
        """
        # Add small epsilon to avoid log(0)
        log_depth = np.log1p(depth_arr - np.min(depth_arr))
        normalized_log_depth = (log_depth - np.min(log_depth)) / (np.max(log_depth) - np.min(log_depth))

        # Use a perceptually uniform colormap for better distinction
        depth_rgb = plt.cm.plasma(normalized_log_depth) * 255
        depth_rgb = depth_rgb.astype(np.uint8)
        return depth_rgb[:, :, :3]