from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration

from datasets import load_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_PARQUET_PATH

import torch

import genesis as gs
from genesis.engine.entities import RigidEntity

from PIL import Image

from dataclasses import dataclass

import os
import itertools

from importlib.resources import files

@dataclass
class HoldState:
    pick_frame_id: int
    place_frame_id: int

@dataclass
class InitialState:
    ball_x: int
    ball_y: int
    cup_x: int
    cup_y: int

@dataclass
class MotorRangeInput:
    motor_id: int
    start: float
    end: float

@dataclass
class GridSearchInput:
    motors: list[MotorRangeInput]  # List of MotorRangeInput
    step: float = 0.01  # Step size for all motors

class EpisodeReplayer:
    LOGGER = Configuration.logger(__name__)

    GRIPPER_ID = 5 # the id of the jaw joint

    MIDDLE_POS_OFFSET = torch.tensor([0, 0.07, 0.06, 0, 1.57, 0.04]) # readjust the middle position calibration

    FIXED_JAW_TRANSLATE = torch.tensor([-2e-2, -9e-2, 0]) # the translation vector from the fixed jaw position to the ball position, in the frame relative to the link
    GOLF_BALL_RADIUS = 4.27e-2 / 2

    DELAY_FRAMES = 4 # the number of fps the follower takes to reflect the leader position

    DIFF_THRESHOLD = 10 # the cutoff value to identify when the gripper is holding the ball and when it is releasing the ball

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]

        # FPS
        self.ds_meta = LeRobotDatasetMetadata(self.repo_id)
        kwargs["fps"] = self.ds_meta.fps
        kwargs["should_start"] = False
        self.show_viewer = kwargs.get("show_viewer", True)

        # Image Resolution of the 1st camera
        camera_key = self.ds_meta.camera_keys[0]
        video_height, video_width, channels = self.ds_meta.features[camera_key]['shape']
        self.res = (video_width, video_height)
        kwargs["res"] = self.res

        self.feetech = Feetech(connect=False)

        self.arm = SoArm100(**kwargs)

        n_envs = kwargs.get("n_envs", None)
        if n_envs is None:
            n_envs = self.ds_meta.total_episodes

        self.build_scene(n_envs=n_envs)

    def load_episodes(self, episode_ids = None):
        self.episode_ids = episode_ids
        if self.episode_ids is None:
            self.episode_count = self.ds_meta.total_episodes
            self.episode_ids = range(self.episode_count)
        else:
            self.episode_count = len(self.episode_ids)

        episode_datasets = [
            self.load_dataset(episode_id)
            for episode_id in self.episode_ids
        ]

        dataloaders = [
            self.get_dataloader(episode_dataset)
            for episode_dataset in episode_datasets
        ]

        self.episodes = [
            next(iter(dataloader))
            for dataloader in dataloaders
        ]

        self.hold_states = [
            self.get_hold_state(episode)
            for episode in self.episodes
        ]

        self.episode_frame_count = min([
            episode_dataset.num_rows
            for episode_dataset in episode_datasets
        ])

    def replay_episodes(self):
        success = self.replay_episode_batch()

        # Log failed episodes
        failed_episode_ids = [self.episode_ids[i] for i in range(len(success)) if not success[i]]
        EpisodeReplayer.LOGGER.info(f"Failed episodes: {','.join(map(str, failed_episode_ids))}")

        score = sum(success) / self.episode_count

        return score

    def replay_episode_batch(self):
        self.set_object_initial_positions()

        for frame_id in range(self.episode_frame_count):
            self.replay_frame(frame_id)

        golf_ball_pos = self.golf_ball.get_pos()
        cup_pos = self.cup.get_pos()

         # project error in the XY plane
        golf_ball_pos_xy = golf_ball_pos[:, :2]
        cup_pos_xy = cup_pos[:, :2]

        distances = torch.norm(golf_ball_pos_xy - cup_pos_xy, dim=1)

        distance_threshold = 0.01
        successes = distances < distance_threshold

        return successes

    def set_object_initial_positions(self):
        # compute the initial positions of the ball and the cup
        initial_states = self.get_initial_states(self.episodes, self.hold_states)

        golf_pos = [
            [initial_state.ball_x, initial_state.ball_y, self.GOLF_BALL_RADIUS]
            for initial_state in initial_states
        ]
        self.golf_ball.set_pos(golf_pos)

        cup_pos = [
            [initial_state.cup_x, initial_state.cup_y, 0]
            for initial_state in initial_states
        ]
        self.cup.set_pos(cup_pos)

    def stop(self):
        self.arm.genesis.stop()

    def load_dataset(self, episode_id):
        episode_chunk = 0
        data_file = DEFAULT_PARQUET_PATH.format(episode_chunk=episode_chunk, episode_index=episode_id)

        dataset = load_dataset(self.repo_id, data_files=[data_file], split="train")
        dataset = dataset.select_columns(["action", "observation.state"])
        return dataset

    def get_dataloader(self, episode_dataset):
        return torch.utils.data.DataLoader(
            episode_dataset,
            batch_size=episode_dataset.num_rows,
        )

    def write_episodes_images(self):
        episode_count = self.ds_meta.total_episodes
        for episode_id in range(episode_count):
            self.write_episode_images(episode_id)

    def write_episode_images(self, episode_id):
        episode_dataset = self.load_dataset(episode_id)

        dataloader = self.get_dataloader(episode_dataset)

        episode = next(iter(dataloader))

        episode_frame_count = episode_dataset.num_rows
        for frame_id in range(episode_frame_count):
            self.write_camera_image(episode, episode_id, frame_id)

    def build_scene(self, n_envs):
        self.arm.genesis.start()

        golf_ball = gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=self.GOLF_BALL_RADIUS,
            pos=(0.25, 0, self.GOLF_BALL_RADIUS)
        )

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup = gs.morphs.Mesh(
            file=cup_filename,
            pos=(-0.25, 0, 0)
        )

        self.golf_ball : RigidEntity = self.arm.genesis.scene.add_entity(
            golf_ball,
            visualize_contact=False, # True
        )

        self.cup : RigidEntity = self.arm.genesis.scene.add_entity(cup)

        self.arm.genesis.build(n_envs=n_envs)
        self.qpos_limits = self.arm.genesis.entity.get_dofs_limit()

    def replay_frame(self, frame_id):
        frame_ids = [
            frame_id
            for _ in range(len(self.episodes))
        ]
        robot_states = self.get_robot_states(self.episodes, frame_ids)

        #EpisodeReplayer.LOGGER.info(f"frame_id = {frame_id}")

        if frame_id == 0:
            self.arm.genesis.entity.set_qpos(robot_states)
        else:
            self.arm.genesis.entity.control_dofs_position(robot_states)

        if self.show_viewer:
            for episode, episode_id in zip(self.episodes, self.episode_ids):
                pass
                #self.write_camera_image(episode, episode_id, frame_id)

        self.arm.genesis.step()

    def get_robot_states(self, episodes, frame_ids):
        robot_states = [
            self.get_robot_state(episode, frame_id)
            for episode, frame_id in zip(episodes, frame_ids)
        ]

        return torch.stack(robot_states)

    def get_robot_state(self, episode, frame_id):
        robot_state = [
            episode['observation.state'][joint_id][frame_id]
            for joint_id in range(Configuration.DOFS)
        ]

        return self.positions_to_radians(robot_state)

    def positions_to_radians(self, positions):
        radians = self.feetech.sim_positions(positions)
        radians = torch.tensor(radians)

        radians = radians + EpisodeReplayer.MIDDLE_POS_OFFSET.to(radians.device)
        radians = torch.clamp(radians, self.qpos_limits[0], self.qpos_limits[1])
        return radians

    def get_hold_state(self, episode) -> HoldState:
        follower_gripper = episode['action'][EpisodeReplayer.GRIPPER_ID]
        leader_gripper = episode['observation.state'][EpisodeReplayer.GRIPPER_ID]

        truncated_leader = leader_gripper[EpisodeReplayer.DELAY_FRAMES:]
        gripper_diff = truncated_leader - follower_gripper[:-EpisodeReplayer.DELAY_FRAMES]

        above_threshold = torch.where(gripper_diff > EpisodeReplayer.DIFF_THRESHOLD, 1, 0)
        return self.sustained_frame_range(above_threshold)

    def sustained_frame_range(self, above_threshold):
        sustained_frames = self.ds_meta.fps # at least 1 sec of holding

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

        hold_start_frames, hold_end_frames
        if len(hold_start_frames) != 1:
            raise Exception("Holding period detection failed")

        return HoldState(pick_frame_id=hold_start_frames[0], place_frame_id=hold_end_frames[0])

    def get_initial_states(self, episodes, hold_states: list[HoldState]) -> list[InitialState]:
        pick_frame_ids = [
            hold_state.pick_frame_id
            for hold_state in hold_states
        ]

        self.set_robot_states(episodes, pick_frame_ids)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, EpisodeReplayer.FIXED_JAW_TRANSLATE)

        place_frame_ids = [
            hold_state.place_frame_id
            for hold_state in hold_states
        ]
        self.set_robot_states(episodes, place_frame_ids)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, EpisodeReplayer.FIXED_JAW_TRANSLATE)

        return [
            InitialState(ball_x=pick_link_pos_i[0].item(), ball_y=pick_link_pos_i[1].item(), cup_x=place_link_pos_i[0].item(), cup_y=place_link_pos_i[1].item())
            for pick_link_pos_i, place_link_pos_i in zip(pick_link_pos, place_link_pos)
        ]

    def set_robot_states(self, episodes, frame_ids):
        robot_states = self.get_robot_states(episodes, frame_ids)
        self.arm.genesis.entity.set_qpos(robot_states)

    def write_image(self, type, rgb_image, episode_id, step_id):
        image = Image.fromarray(rgb_image, mode='RGB')

        image_path = f"img/{self.repo_id}/{type}/episode_{episode_id:03d}/frame_{step_id:03d}.png"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        image.save(image_path)

    def write_camera_image(self, episode, episode_id, frame_id):
        camera_key = self.ds_meta.camera_keys[0]
        camera_image = episode[camera_key][frame_id]
        camera_image = camera_image.data.numpy()
        camera_image = camera_image.transpose(1, 2, 0)

        # convert from [0-1] floats to [0-256[ ints
        camera_image = (camera_image * 255).astype("uint8")

        self.write_image("real", camera_image, episode_id, frame_id)

    def grid_search(self, grid_input: GridSearchInput):
        """
        Grid search to optimize offsets in MIDDLE_POS_OFFSET for the specified motors and ranges.
        Args:
            grid_input (GridSearchInput): Contains motors (MotorRangeInput) and step size.
        Returns:
            Tuple of best offsets (in the order of grid_input.motors)
        """

        best_score = float('-inf')
        best_offsets = None
        # Generate value ranges for each motor
        value_ranges = [
            torch.arange(motor.start, motor.end + grid_input.step, grid_input.step)
            for motor in grid_input.motors
        ]
        for offsets in itertools.product(*value_ranges):
            # Update in place
            for idx, motor in enumerate(grid_input.motors):
                EpisodeReplayer.MIDDLE_POS_OFFSET[motor.motor_id] = offsets[idx].item()
            score = self.replay_episodes()
            EpisodeReplayer.LOGGER.info(f"Score: {score:.4f} for {EpisodeReplayer.MIDDLE_POS_OFFSET}")
            if score > best_score:
                best_score = score
                best_offsets = tuple(offset.item() for offset in offsets)
        return best_offsets
