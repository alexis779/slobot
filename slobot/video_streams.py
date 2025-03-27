import time
import queue
import os
import threading
import numpy as np
import matplotlib.pyplot as plt


from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100


class VideoStreams:
    FRAME_TYPES = ["rgb", "depth", "segmentation", "normal"]

    def __init__(self):
        os.makedirs(Configuration.WORK_DIR, exist_ok=True)

    def frame_filenames(self, fps, res):
        cam_id = 0
        env_id = 0
        filenames = self.start(cam_id, [env_id], None, res, fps=fps, rgb=True, depth=True, segmentation=True, normal=True)
        filenames = list(map(lambda filename: filename[env_id], filenames))

        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path, frame_handler=self, res=res, show_viewer=False)
        arm.elemental_rotations()

        self.stop()

        # stop genesis
        arm.genesis.stop()
        return filenames

    def start(
        self,
        cam_id,
        env_ids,
        save_to_filename,
        res,
        fps,
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False
    ):
        self.env_ids = env_ids

        self.period = 1.0 / fps

        self.frame_enabled = [rgb, depth, segmentation, normal]

        self.save_to_filename = save_to_filename

        date_time = time.strftime('%Y%m%d_%H%M%S')

        filenames = [
            [
                self._filename(cam_id, env_id, self.FRAME_TYPES[frame_id], date_time)
                for env_id in self.env_ids
            ] if self.frame_enabled[frame_id] else None
            for frame_id in range(len(self.FRAME_TYPES))
        ]

        self.writers = [
            [
                self._writer(filenames[frame_id][env_id], res, fps)
                for env_id in self.env_ids
            ] if self.frame_enabled[frame_id] else None
            for frame_id in range(len(self.FRAME_TYPES))
        ]

        self.start_time = time.time()
        self.duration = 0
        self.previous_frame = None

        self.queue = queue.Queue()

        # Start the poller
        self.poller = threading.Thread(target=self.poll)
        self.poller.start()

        return filenames

    def poll(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break

            for frame_id in range(len(self.FRAME_TYPES)):
                typed_frame = frame[frame_id]
                if typed_frame is None:
                    #if self.frame_enabled[frame_id]:
                    #    print(f"Frame type {self.FRAME_TYPES[frame_id]} is enabled but not provided")
                    continue

                if self.writers[frame_id] is None:
                    continue

                for env_id in self.env_ids:
                    env_arr = typed_frame if len(self.env_ids) == 1 else typed_frame[env_id]

                    writer = self.writers[frame_id][env_id]
                    writer.write_frame(env_arr)

    def handle_frame(self, frame):
        current_time = time.time()

        rgb_arr, depth_arr, seg_arr, normal_arr = frame

        if depth_arr is not None:
            depth_arr = self._logarithmic_depth_to_rgb(depth_arr)

        current_frame = (rgb_arr, depth_arr, seg_arr, normal_arr)

        for frame_id in range(len(self.FRAME_TYPES)):
            typed_frame = current_frame[frame_id]
            if typed_frame is None:
                if self.frame_enabled[frame_id]:
                    print(f"Frame type {self.FRAME_TYPES[frame_id]} is enabled but not provided")

        if self.previous_frame is None:
            self.previous_frame = current_frame

        while self.start_time + self.duration < current_time:
            self.duration += self.period
            self.queue.put(self.previous_frame)

        self.previous_frame = current_frame

    def stop(self):
        if self.previous_frame is not None:
            current_time = time.time()
            while self.start_time + self.duration < current_time:
                self.duration += self.period
                self.queue.put(self.previous_frame)

        self.queue.put(None) # send poison pill
        self.poller.join()

        for frame_id in range(len(self.FRAME_TYPES)):
            for env_id in self.env_ids:
                if self.writers[frame_id] is None:
                    continue

                writer = self.writers[frame_id][env_id]
                if writer is None:
                    continue

                writer.close()

    def _writer(self, filename, res, fps):
        return FFMPEG_VideoWriter(
            filename,
            res,
            fps,
        )

    def _filename(self, cam_id, env_id, frame_type, date_time):
        if self.save_to_filename is not None and env_id == 0 and len(self.env_ids) == 1 and frame_type == 'rgb':
            return self.save_to_filename

        return f"{Configuration.WORK_DIR}/cam_{cam_id}_env_{env_id}_{frame_type}_{date_time}.mp4"


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