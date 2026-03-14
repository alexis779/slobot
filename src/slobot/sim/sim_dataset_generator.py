import os

import torch

from slobot.sim.sim_policy import SimPolicy
from slobot.sim.recording_layout import RecordingLayout
from slobot.sim.recording_layout import PreGraspMode
from slobot.sim.data_sampler import DataSampler
from slobot.configuration import Configuration


# Default ranges for ball_x, ball_y, cup_x, cup_y (in cm)
DEFAULT_LAYOUT_RANGES = [
    (-10.0, 10.0),   # ball_x
    (-15.0, -5.0),   # ball_y
    (-10.0, 10.0),   # cup_x
    (-15.0, -5.0),   # cup_y
]


class SimDatasetGenerator:
    LOGGER = Configuration.logger(__name__)

    def __init__(self, **kwargs):
        self.episode_count = kwargs["episode_count"]

        self.data_sampler = DataSampler(
            ranges=DEFAULT_LAYOUT_RANGES,
            seed=0,
        )
        self.sim_policy = SimPolicy()
        self.pre_grasp_mode = PreGraspMode.VERTICAL

    def generate_dataset(self):
        successful = []
        failed = []
        sample_iter = iter(self.data_sampler)

        while len(successful) < self.episode_count:
            attempt = len(successful) + len(failed)
            sample = next(sample_iter)
            recording_id = f"episode{attempt:03d}"

            try:
                self._generate_episode(recording_id, sample)
                successful.append(recording_id)
            except Exception as e:
                self.LOGGER.warning("Skipping episode %s: %s", recording_id, e)
                failed.append(recording_id)

        self._cleanup_recordings(successful, failed)

    def _generate_episode(self, recording_id: str, sample: torch.Tensor):
        ball_x, ball_y, cup_x, cup_y = sample
        recording_layout = RecordingLayout(
            rrd_file=None,
            pick_frame_id=None,
            pre_grasp_mode=self.pre_grasp_mode,
            ball_x=ball_x,
            ball_y=ball_y,
            cup_x=cup_x,
            cup_y=cup_y,
            recording_id=recording_id,
        )
        self.sim_policy.execute(recording_layout)

    def _cleanup_recordings(self, successful: list[str], failed: list[str]):
        for recording_id in failed:
            path = self.sim_policy.rerun_metrics.recording_path(recording_id)
            if os.path.exists(path):
                os.remove(path)
                self.LOGGER.info("Deleted failed recording %s", path)

        # Rename successful recordings to consecutive indices
        for new_idx, recording_id in enumerate(successful):
            old_path = self.sim_policy.rerun_metrics.recording_path(recording_id)
            new_recording_id = f"episode{new_idx:03d}"
            new_path = self.sim_policy.rerun_metrics.recording_path(new_recording_id)
            if old_path != new_path:
                os.rename(old_path, new_path)
                self.LOGGER.info("Renamed %s -> %s", old_path, new_path)
