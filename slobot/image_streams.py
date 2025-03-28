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

    def frame_filenames(self, res, fps):
        thread = threading.Thread(target=self.run_simulation, args=(res, fps))
        thread.start()

        while True:
            simulation_frame_paths = self.queue.get()
            if simulation_frame_paths is None:
                break

            yield simulation_frame_paths

        thread.join()

    def run_simulation(self, res, fps):
        self.start(res, fps, rgb=True, depth=True, segmentation=True, normal=True)

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, frame_handler=self, res=res, show_viewer=False)
        arm.elemental_rotations()

        self.stop()

        # stop genesis
        arm.genesis.stop()

    def start(
        self,
        res,
        fps,
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False
    ):
        self.simulation_frames = []

        self.res = res
        self.fps = fps
        self.period = 1.0 / fps
        self.segment_id = 0
        self.video_timestamp = time.time()
        self.current_frame = None

        self.frame_enabled = [rgb, depth, segmentation, normal]

    def stop(self):
        if len(self.simulation_frames) > 0:
            self.flush_frames()
        
        self.queue.put(None) # add poison pill

    def handle_frame(self, frame):
        timestamp = time.time()

        rgb_arr, depth_arr, seg_arr, normal_arr = frame

        if depth_arr is not None:
           depth_arr = VideoStreams.logarithmic_depth_to_rgb(depth_arr)

        simulation_frame = SimulationFrame(timestamp, rgb_arr, depth_arr, seg_arr, normal_arr)

        self.simulation_frames.append(simulation_frame)
        if self.video_timestamp + self.period <= simulation_frame.timestamp:
            self.flush_frames()

    def flush_frames(self):
        simulation_frame_paths = self.transcode_frames()
        self.queue.put(simulation_frame_paths)

    def transcode_frames(self) -> SimulationFramePaths:
        date_time = time.strftime('%Y%m%d_%H%M%S')
        if self.current_frame is None:
            self.current_frame = self.simulation_frames[0]

        for simulation_frame in self.simulation_frames:
            while self.video_timestamp + self.period < simulation_frame.timestamp:
                self.video_timestamp += self.period

            self.current_frame = simulation_frame

        simulation_frame_images = []
        for frame_id in range(len(VideoStreams.FRAME_TYPES)):
            if not self.frame_enabled[frame_id]:
                next

            filename = self._filename(VideoStreams.FRAME_TYPES[frame_id], date_time, self.segment_id)
            self.create_image_paths(self.current_frame.frame(frame_id), filename)
            simulation_frame_images.append(filename)

        self.simulation_frames.clear()
        self.segment_id += 1

        return SimulationFramePaths(self.current_frame.timestamp, simulation_frame_images)

    def create_image_paths(self, typed_array, filename):
        image = Image.fromarray(typed_array, mode='RGB')
        image.save(filename)
        return filename

    def _filename(self, frame_type, date_time, segment_id):
        return f"{Configuration.WORK_DIR}/{frame_type}_{date_time}_{segment_id}.webp"

