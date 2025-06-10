from slobot.so_arm_100 import SoArm100
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import torch

class EpisodeReplayer:
    SIGN = torch.tensor([-1, -1, 1, 1, -1, 1])

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]
        ds_meta = LeRobotDatasetMetadata(self.repo_id)
        kwargs["fps"] = ds_meta.fps
        self.arm = SoArm100(**kwargs)
        self.qpos_limits = self.arm.entity.get_dofs_limit()

    def replay_episode(self, episode_index):
        dataset = LeRobotDataset(self.repo_id, episodes=[episode_index])

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()
        for idx in range(from_idx, to_idx):
            frame = dataset[idx]
            self.replay_frame(frame)

        self.arm.genesis.hold_entity()
    
    def replay_frame(self, frame):
        action = frame["action"]
        qpos = self.degrees_to_radians(action)
        self.arm.entity.set_qpos(qpos)
        self.arm.genesis.step()
    
    def degrees_to_radians(self, degrees):
        radians = torch.deg2rad(degrees)
        radians = radians * self.SIGN
        radians = torch.clamp(radians, self.qpos_limits[0], self.qpos_limits[1])
        return radians