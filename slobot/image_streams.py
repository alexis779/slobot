import time
import os
import threading
import queue

from PIL import Image

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.video_streams import VideoStreams
from slobot.simulation_frame import SimulationFrame
from slobot.simulation_frame_paths import SimulationFramePaths

# Generate a stream of images from the simulation
class ImageStreams:

    def __init__(self):
        os.makedirs(Configuration.WORK_DIR, exist_ok=True)
        self.queue = queue.Queue()

    def simulation_frame_paths(self, res, fps, rgb=True, depth=False, segmentation=False, normal=False):
        thread = threading.Thread(target=self.run_simulation, args=(res, fps, rgb, depth, segmentation, normal))
        thread.start()

        while True:
            simulation_frame_paths = self.queue.get()
            if simulation_frame_paths is None:
                break

            yield simulation_frame_paths

        thread.join()

    def run_simulation(self, res, fps, rgb=True, depth=False, segmentation=False, normal=False):
        self.start(res, fps, rgb=rgb, depth=depth, segmentation=segmentation, normal=normal)

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, step_handler=self, res=res, fps=fps, show_viewer=False, rgb=rgb, depth=depth, segmentation=segmentation, normal=normal)
        arm.elemental_rotations()
        arm.stop()

        self.queue.put(None) # add poison pill

    def start(
        self,
        res,
        fps,
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False
    ):
        self.res = res
        self.fps = fps
        self.segment_id = 0

        self.frame_enabled = [rgb, depth, segmentation, normal]

    def handle_step(self, simulation_frame: SimulationFrame):
        if simulation_frame.depth is not None:
           # colorize depth
           simulation_frame.depth = VideoStreams.logarithmic_depth_to_rgb(simulation_frame.depth)

        simulation_frame_paths = self.transcode_frame(simulation_frame)
        self.queue.put(simulation_frame_paths)

    def transcode_frame(self, simulation_frame: SimulationFrame) -> SimulationFramePaths:
        date_time = time.strftime('%Y%m%d_%H%M%S')

        simulation_frame_images = []
        for frame_id in range(len(VideoStreams.FRAME_TYPES)):
            if not self.frame_enabled[frame_id]:
                continue

            filename = self._filename(VideoStreams.FRAME_TYPES[frame_id], date_time, self.segment_id)
            self.create_image_paths(simulation_frame.frame(frame_id), filename)
            simulation_frame_images.append(filename)

        self.segment_id += 1

        return SimulationFramePaths(simulation_frame=simulation_frame, paths=simulation_frame_images)

    def create_image_paths(self, typed_array, filename):
        image = Image.fromarray(typed_array, mode='RGB')
        image.save(filename)
        return filename

    def _filename(self, frame_type, date_time, segment_id):
        return f"{Configuration.WORK_DIR}/{frame_type}_{date_time}_{segment_id}.webp"

