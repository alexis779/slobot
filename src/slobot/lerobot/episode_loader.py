import torch
from dataclasses import dataclass


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from slobot.feetech import Feetech
from slobot.configuration import Configuration

@dataclass
class HoldState:
    pick_frame_id: int
    place_frame_id: int


class EpisodeLoader:
    COLUMN_NAMES = [ 'frame_index', 'action', 'observation.state' ]

    LEADER_STATE_COLUMN = 'action'
    FOLLOWER_STATE_COLUMN = 'observation.state'

    DELAY_FRAMES = 4 # the number of fps the follower takes to reflect the leader position
    DIFF_THRESHOLD = 10 # the cutoff value to identify when the gripper is holding the ball and when it is releasing the ball

    def __init__(self, repo_id, device: torch.device):
        self.repo_id = repo_id
        self.dataset = LeRobotDataset(repo_id=self.repo_id)
        self.feetech = Feetech(connect=False)

    def load_episodes(self, episode_ids = None):
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

        self.hold_states = [
            self.get_hold_state(episode)
            for episode in self.episodes
        ]

        self.episode_frame_count = min([
            len(episode['frame_index'])
            for episode in self.episodes
        ])

    def get_hold_state(self, episode) -> HoldState:
        follower_gripper = episode['action'][:, Configuration.GRIPPER_ID]
        leader_gripper = episode['observation.state'][:, Configuration.GRIPPER_ID]

        truncated_leader = leader_gripper[EpisodeLoader.DELAY_FRAMES:]
        gripper_diff = truncated_leader - follower_gripper[:-EpisodeLoader.DELAY_FRAMES]

        above_threshold = torch.where(gripper_diff > EpisodeLoader.DIFF_THRESHOLD, 1, 0)
        return self.sustained_frame_range(above_threshold)

    def sustained_frame_range(self, above_threshold):
        sustained_frames = self.dataset.meta.fps # at least 1 sec of holding

        counter = torch.full_like(above_threshold, fill_value=0)

        frame = len(above_threshold) - 1
        counter[frame] = 1 if above_threshold[frame] == 1 else 0

        hold_start_frames = []
        hold_end_frames = []

        for frame in range(frame-1, -1, -1):
            if above_threshold[frame] == 1:
                counter[frame] = counter[frame+1] + 1
            else:
                if counter[frame+1] >= sustained_frames:
                    hold_start_frames.append(frame+1)
                    hold_end_frame = frame + counter[frame+1]
                    hold_end_frame = hold_end_frame.item()
                    hold_end_frames.append(hold_end_frame)

                counter[frame] = 0

        frame = 0
        if counter[frame] >= sustained_frames:
            hold_start_frames.append(frame)
            hold_end_frame = frame + counter[frame] - 1
            hold_end_frame = hold_end_frame.item()
            hold_end_frames.append(hold_end_frame)

        if len(hold_start_frames) != 1:
            raise Exception("Holding period detection failed")

        return HoldState(pick_frame_id=hold_start_frames[0], place_frame_id=hold_end_frames[0])

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

    def set_middle_pos_offset(self, middle_pos_offset):
        self.middle_pos_offset = middle_pos_offset

    def set_dofs_limit(self, dofs_limit):
        self.dofs_limit = dofs_limit

    def positions_to_radians(self, positions):
        radians = self.feetech.sim_positions(positions)
        radians = torch.tensor(radians, device=self.middle_pos_offset.device)

        radians = radians + self.middle_pos_offset
        radians = torch.clamp(radians, self.dofs_limit[0], self.dofs_limit[1])
        return radians

