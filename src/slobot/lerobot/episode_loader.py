import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from slobot.feetech import Feetech
from slobot.configuration import Configuration
from slobot.lerobot.frame_delay_detector import FrameDelayDetector
from slobot.lerobot.hold_state_detector import HoldStateDetector, HoldState

class EpisodeLoader:
    COLUMN_NAMES = [ 'frame_index', 'action', 'observation.state' ]
    DIFF_THRESHOLD = 10

    LEADER_STATE_COLUMN = 'action'
    FOLLOWER_STATE_COLUMN = 'observation.state'

    def __init__(self, repo_id, episode_ids):
        self.repo_id = repo_id
        self.load_episodes(episode_ids)
        self.feetech = Feetech(connect=False)
        self.middle_pos_offset = torch.tensor([0,  0.1,  0.1,  0,  torch.pi/2,  0])

    def load_episodes(self, episode_ids):
        self.episode_ids = episode_ids

        self.dataset = LeRobotDataset(repo_id=self.repo_id, episodes=episode_ids)

        self.episode_ids = episode_ids
        if self.episode_ids is None:
            self.episode_count = self.dataset.meta.total_episodes
            self.episode_ids = range(self.episode_count)
            self.episode_indexes = {
                episode_id: episode_id
                for episode_id in self.episode_ids
            }
        else:
            self.episode_count = len(self.episode_ids)
            self.episode_indexes = {
                self.episode_ids[episode_id]: episode_id
                for episode_id in range(self.episode_count)
            }

        self.episodes = [
            {
                column_name: []
                for column_name in EpisodeLoader.COLUMN_NAMES
            }
            for episode_id in range(self.episode_count)
        ]

        for row in self.dataset.hf_dataset:
            episode_id = row['episode_index'].item()
            if episode_id not in self.episode_indexes:
                continue

            episode_index = self.episode_indexes[episode_id]
            for column_name in EpisodeLoader.COLUMN_NAMES:
                self.episodes[episode_index][column_name].append(row[column_name])

        for episode_id in self.episode_ids:
            episode_index = self.episode_indexes[episode_id]
            episode = self.episodes[episode_index]
            for column_name in EpisodeLoader.COLUMN_NAMES:
                episode[column_name] = torch.vstack(episode[column_name])

        self.episode_frame_count = min([
            len(episode['frame_index'])
            for episode in self.episodes
        ])

        self.hold_states = [
            self.get_hold_state(episode)
            for episode in self.episodes
        ]

    def get_hold_state(self, episode) -> HoldState:
        leader_gripper = episode['action'][:, Configuration.GRIPPER_ID]
        follower_gripper = episode['observation.state'][:, Configuration.GRIPPER_ID]

        frame_delay_detector = FrameDelayDetector(fps=self.dataset.meta.fps)
        delay_frames = frame_delay_detector.detect_frame_delay(leader_gripper, follower_gripper)

        leader_gripper = leader_gripper[:-delay_frames]
        follower_gripper = follower_gripper[delay_frames:]

        hold_state_detector = HoldStateDetector(diff_threshold=EpisodeLoader.DIFF_THRESHOLD)
        hold_state_detector.replay_teleop(leader_gripper, follower_gripper)

        hold_state = hold_state_detector.get_hold_state()
        if hold_state.pick_frame_id is None or hold_state.place_frame_id is None:
            raise ValueError("Hold state not found")

        return hold_state

    def get_robot_states(self, column_name, frame_ids):
        robot_states = [
            self.get_robot_state(episode, frame_id, column_name)
            for episode, frame_id in zip(self.episodes, frame_ids)
        ]

        return torch.stack(robot_states)

    def get_robot_state(self, episode, frame_id, column_name):
        robot_state = [
            episode[column_name][frame_id][joint_id]
            for joint_id in range(Configuration.DOFS)
        ]

        return self.positions_to_radians(robot_state)

    def set_middle_pos_offset(self, middle_pos_offset: torch.Tensor):
        self.middle_pos_offset = middle_pos_offset

    def set_dofs_limit(self, dofs_limit):
        self.dofs_limit = dofs_limit

    def positions_to_radians(self, positions):
        radians = self.feetech.sim_positions(positions)
        radians = torch.tensor(radians, device=self.middle_pos_offset.device)

        radians = radians + self.middle_pos_offset
        radians = torch.clamp(radians, self.dofs_limit[0], self.dofs_limit[1])
        return radians

