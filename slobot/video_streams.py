import time
import os
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.simulation_frame import SimulationFrame
from slobot.simulation_frame_paths import SimulationFramePaths
from slobot.video_writer import VideoWriter


class VideoStreams:
    FRAME_TYPES = ["rgb", "depth", "segmentation", "normal"]

    def __init__(self, **kwargs):
        os.makedirs(Configuration.WORK_DIR, exist_ok=True)
        self.logger = Configuration.logger(__name__)

        self.video_segment_queue = queue.Queue()
        self.transcode_queue = queue.Queue()

        self.codec = kwargs.get('codec', 'libx264')

    def frame_filenames(self, res, fps, segment_duration):
        # run simulation in a separate thread
        thread = threading.Thread(target=self.run_simulation, args=(res, fps, segment_duration))
        thread.start()

        while True:
            simulation_frame_paths = self.video_segment_queue.get()
            if simulation_frame_paths is None:
                break

            yield simulation_frame_paths

        thread.join()

    def run_simulation(self, res, fps, segment_duration):
        cam_id = 0
        env_id = 0
        self.start(cam_id, [env_id], res, fps, segment_duration, rgb=True, depth=True, segmentation=True, normal=True)

        # run transcoding in a separate thread
        self.transcode_thread = threading.Thread(target=self.poll_transcode)
        self.transcode_thread.start()

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, frame_handler=self, res=res, show_viewer=False)
        arm.elemental_rotations()

        self.stop()

        # stop genesis
        arm.genesis.stop()

    def start(
        self,
        cam_id,
        env_ids,
        res,
        fps,
        segment_duration,
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False
    ):
        self.cam_id = cam_id
        self.env_ids = env_ids
        self.res = res
        self.fps = fps

        self.period = 1.0 / self.fps
        self.segment_id = 0

        self.frame_enabled = [rgb, depth, segmentation, normal]

        self.video_timestamp = time.time()
        self.current_frame = None

        self.segment_duration = segment_duration

        self.simulation_frames = []

    def handle_frame(self, frame):
        timestamp = time.time()

        rgb_arr, depth_arr, seg_arr, normal_arr = frame

        if depth_arr is not None:
            depth_arr = VideoStreams.logarithmic_depth_to_rgb(depth_arr)

        simulation_frame = SimulationFrame(timestamp, rgb_arr, depth_arr, seg_arr, normal_arr)

        self.simulation_frames.append(simulation_frame)

        if self.video_timestamp + self.segment_duration <= simulation_frame.timestamp:
            self.flush_frames()

    def flush_frames(self):
        self.transcode_queue.put(self.simulation_frames)
        self.simulation_frames = []

    def poll_transcode(self):
        while True:
            simulation_frames = self.transcode_queue.get()
            if simulation_frames is None:
                break

            simulation_frame_paths = self.transcode_frames(simulation_frames)
            self.video_segment_queue.put(simulation_frame_paths)

    def transcode_frames(self, simulation_frames) -> SimulationFramePaths:
        self.logger.info(f"Transcoding video segment {self.segment_id}")
        date_time = time.strftime('%Y%m%d_%H%M%S')

        if self.current_frame is None:
            self.current_frame = simulation_frames[0]

        first_timestamp = simulation_frames[0].timestamp

        video_frames = []
        for simulation_frame in simulation_frames:
            while self.video_timestamp + self.period < simulation_frame.timestamp:
                video_frames.append(self.current_frame)
                self.video_timestamp += self.period

            # TODO last frame of the simulation will be dropped
            self.current_frame = simulation_frame

        simulation_frames.clear()

        simulation_frame_videos = []
        for env_id in self.env_ids:
            env_simulation_frame_videos = []
            for frame_id in range(len(self.FRAME_TYPES)):
                if not self.frame_enabled[frame_id]:
                    next

                filename = self._filename(self.cam_id, env_id, self.FRAME_TYPES[frame_id], date_time, self.segment_id)

                video_writer = VideoWriter(filename, self.res, self.fps, codec=self.codec)
                type_frames = [
                    simulation_frame.frame(frame_id)
                    for simulation_frame in video_frames
                ]
                if len(self.env_ids) > 1:
                    type_frames = [
                        env_frame[env_id]
                        for env_frame in type_frames
                    ]
                type_frames = np.array(type_frames)
                video_writer.transcode(type_frames, filename)

                env_simulation_frame_videos.append(filename)
            simulation_frame_videos.append(env_simulation_frame_videos)

        self.logger.info(f"Done flushing frames for segment {self.segment_id}")
        self.segment_id += 1

        return SimulationFramePaths(first_timestamp, simulation_frame_videos)

    def stop(self):
        self.flush_frames()
        self.transcode_queue.put(None) # add poison pill
        self.transcode_thread.join()

        self.video_segment_queue.put(None) # add poison pill

    def _filename(self, cam_id, env_id, frame_type, date_time, segment_id):
        return f"{Configuration.WORK_DIR}/cam_{cam_id}_env_{env_id}_{frame_type}_{date_time}_{segment_id}.ts"

    def logarithmic_depth_to_rgb(depth_arr):
        """
        Use logarithmic scaling to enhance depth visualization
        Helps spread out colors better than linearly potentially improving contrast
        """
        # Add small epsilon to avoid log(0)
        log_depth = np.log1p(depth_arr - np.min(depth_arr))
        normalized_log_depth = (log_depth - np.min(log_depth)) / (np.max(log_depth) - np.min(log_depth))

        # Use a perceptually uniform colormap for better distinction
        depth_rgb = plt.cm.plasma(normalized_log_depth) * 255
        depth_rgb = depth_rgb.astype(np.uint8)
        return depth_rgb[:, :, :3]