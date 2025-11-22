from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration
from slobot.metrics.metrics import Metrics

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.utils import geom as gu

from PIL import Image

from dataclasses import dataclass

import os

from importlib.resources import files

@dataclass
class HoldState:
    pick_frame_id: int
    place_frame_id: int

@dataclass
class InitialState:
    ball: torch.Tensor  # 3D float tensor [x, y, z]
    cup: torch.Tensor  # 3D float tensor [x, y, z]

class EpisodeReplayer:
    LOGGER = Configuration.logger(__name__)

    COLUMN_NAMES = [ 'frame_index', 'action', 'observation.state' ]

    LEADER_STATE_COLUMN = 'action'
    FOLLOWER_STATE_COLUMN = 'observation.state'

    GRIPPER_ID = 5 # the id of the jaw joint

    MIDDLE_POS_OFFSET = torch.tensor([-0.0097,  0.1134,  0.1031,  0.0426,  1.6127,  0.35]) # readjust the middle position calibration

    FIXED_JAW_TRANSLATE = torch.tensor([-1.4e-2, -9e-2, 0]) # the translation vector from the fixed jaw position to the ball position, in the frame relative to the link
    GOLF_BALL_RADIUS = 4.27e-2 / 2

    DELAY_FRAMES = 4 # the number of fps the follower takes to reflect the leader position

    DIFF_THRESHOLD = 10 # the cutoff value to identify when the gripper is holding the ball and when it is releasing the ball

    DISTANCE_THRESHOLD = 0.01 # the threshold for the distance between the golf ball and the cup for the ball to be considered in the cup, or for the ball to be considered moved from the initial position

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]

        self.dataset = LeRobotDataset(self.repo_id)

        self.middle_pos_offset = EpisodeReplayer.MIDDLE_POS_OFFSET
        self.fixed_jaw_translate = EpisodeReplayer.FIXED_JAW_TRANSLATE
        # FPS
        kwargs["fps"] = self.dataset.meta.fps
        kwargs["should_start"] = False

        kwargs["show_viewer"] = kwargs.get("show_viewer", False)

        # Image Resolution of the 1st camera
        camera_key = self.dataset.meta.camera_keys[0]
        video_height, video_width, channels = self.dataset.meta.features[camera_key]['shape']
        self.res = (video_width, video_height)
        kwargs["res"] = self.res

        self.add_metrics = kwargs.get("add_metrics", False)
        if self.add_metrics:
            self.metrics = Metrics()
            kwargs["step_handler"] = self.metrics

        self.arm = SoArm100(**kwargs)

        self.feetech = Feetech(connect=False)

        n_envs = kwargs.get("n_envs", None)
        if n_envs is None:
            n_envs = self.dataset.meta.total_episodes

        self.build_scene(n_envs=n_envs)

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
                for column_name in EpisodeReplayer.COLUMN_NAMES
            }
            for episode_id in range(self.episode_count)
        ]

        for row in self.dataset.hf_dataset:
            episode_id = row['episode_index'].item()
            if episode_id not in self.episode_indexes:
                continue

            episode_index = self.episode_indexes[episode_id]
            for column_name in EpisodeReplayer.COLUMN_NAMES:
                self.episodes[episode_index][column_name].append(row[column_name])

        for episode_id in self.episode_ids:
            episode_index = self.episode_indexes[episode_id]
            episode = self.episodes[episode_index]
            for column_name in EpisodeReplayer.COLUMN_NAMES:
                episode[column_name] = torch.vstack(episode[column_name])

        self.hold_states = [
            self.get_hold_state(episode)
            for episode in self.episodes
        ]

        self.episode_frame_count = min([
            len(episode['frame_index'])
            for episode in self.episodes
        ])

    def replay_episodes(self):
        moved, success = self.replay_episode_batch()

        # Log failed episodes
        failed_episode_ids = [self.episode_ids[i] for i in range(len(success)) if not success[i]]
        EpisodeReplayer.LOGGER.info(f"Failed episodes: {','.join(map(str, failed_episode_ids))}")

        score = (sum(moved) + sum(success)) / (2 * self.episode_count)

        return score

    def replay_episode_batch(self):
        self.set_object_initial_positions()

        for frame_id in range(self.episode_frame_count):
            self.replay_frame(frame_id)

        initial_golf_ball_pos = torch.stack([
            initial_state.ball[:2]
            for initial_state in self.initial_states
        ])

        golf_ball_pos = self.golf_ball.get_pos()

         # project to the XY plane
        golf_ball_pos = golf_ball_pos[:, :2]

        golf_ball_to_initial = torch.norm(golf_ball_pos - initial_golf_ball_pos, dim=1)

        moved = golf_ball_to_initial > EpisodeReplayer.DISTANCE_THRESHOLD

        cup_pos = self.cup.get_pos()

        cup_pos = cup_pos[:, :2]

        golf_ball_to_cup = torch.norm(golf_ball_pos - cup_pos, dim=1)

        successes = golf_ball_to_cup < EpisodeReplayer.DISTANCE_THRESHOLD

        return moved, successes

    def replay_episode_frames(self):
        [
            self.replay_frame(frame_id)
            for frame_id in range(self.episode_frame_count)
        ]

    def set_object_initial_positions(self):
        # compute the initial positions of the ball and the cup
        self.initial_states = self.get_initial_states()

        golf_pos = [
            [initial_state.ball[0].item(), initial_state.ball[1].item(), self.GOLF_BALL_RADIUS]
            for initial_state in self.initial_states
        ]
        self.golf_ball.set_pos(golf_pos)

        cup_pos = [
            [initial_state.cup[0].item(), initial_state.cup[1].item(), 0]
            for initial_state in self.initial_states
        ]
        self.cup.set_pos(cup_pos)

    def stop(self):
        self.arm.genesis.stop()

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
            vis_mode='visual', # collision
        )

        self.cup : RigidEntity = self.arm.genesis.scene.add_entity(cup)

        self.arm.genesis.build(n_envs=n_envs)
        self.qpos_limits = self.arm.genesis.entity.get_dofs_limit()

    def replay_frame(self, frame_id):
        frame_ids = [
            frame_id
            for _ in range(len(self.episodes))
        ]
        leader_robot_states = self.get_robot_states(self.episodes, frame_ids, EpisodeReplayer.LEADER_STATE_COLUMN)

        #EpisodeReplayer.LOGGER.info(f"frame_id = {frame_id}")

        if frame_id == 0:
            self.arm.genesis.entity.set_dofs_position(leader_robot_states)
        else:
            self.arm.genesis.entity.control_dofs_position(leader_robot_states)

        '''
        if frame_id == self.hold_states[0].pick_frame_id:
            for _ in range(50):
                self.arm.genesis.draw_arrow(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate, self.GOLF_BALL_RADIUS, (1, 0, 0, 0.5))
                self.arm.genesis.step()
                input()
        '''
        self.arm.genesis.step()

        follower_robot_states = self.get_robot_states(self.episodes, frame_ids, EpisodeReplayer.FOLLOWER_STATE_COLUMN)
        if self.add_metrics:
            self.metrics.add_metric("real.qpos", follower_robot_states)

    def get_robot_states(self, episodes, frame_ids, column_name):
        robot_states = [
            self.get_robot_state(episode, frame_id, column_name)
            for episode, frame_id in zip(episodes, frame_ids)
        ]

        return torch.stack(robot_states)

    def get_robot_state(self, episode, frame_id, column_name):
        robot_state = [
            episode[column_name][frame_id][joint_id]
            for joint_id in range(Configuration.DOFS)
        ]

        return self.positions_to_radians(robot_state)

    def positions_to_radians(self, positions):
        radians = self.feetech.sim_positions(positions)
        radians = torch.tensor(radians, device=self.middle_pos_offset.device)

        radians = radians + self.middle_pos_offset
        radians = torch.clamp(radians, self.qpos_limits[0], self.qpos_limits[1])
        return radians

    def get_hold_state(self, episode) -> HoldState:
        follower_gripper = episode['action'][:, EpisodeReplayer.GRIPPER_ID]
        leader_gripper = episode['observation.state'][:, EpisodeReplayer.GRIPPER_ID]

        truncated_leader = leader_gripper[EpisodeReplayer.DELAY_FRAMES:]
        gripper_diff = truncated_leader - follower_gripper[:-EpisodeReplayer.DELAY_FRAMES]

        above_threshold = torch.where(gripper_diff > EpisodeReplayer.DIFF_THRESHOLD, 1, 0)
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

    def get_initial_states(self) -> list[InitialState]:
        pick_frame_ids = [
            hold_state.pick_frame_id
            for hold_state in self.hold_states
        ]

        self.set_robot_states(self.episodes, pick_frame_ids)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, EpisodeReplayer.FIXED_JAW_TRANSLATE)

        place_frame_ids = [
            hold_state.place_frame_id
            for hold_state in self.hold_states
        ]
        self.set_robot_states(self.episodes, place_frame_ids)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, EpisodeReplayer.FIXED_JAW_TRANSLATE)

        return [
            InitialState(
                ball=pick_link_pos_i,
                cup=place_link_pos_i,
            )
            for pick_link_pos_i, place_link_pos_i, in zip(pick_link_pos, place_link_pos)
        ]

    def get_initial_state_images(self, video_episode):
        pick_frame_ids = [
            hold_state.pick_frame_id
            for hold_state in self.hold_states
        ]

        self.set_robot_states(self.episodes, pick_frame_ids)
        self.arm.genesis.step()
        sim_pick_image = self.get_sim_image()
        real_pick_image = self.get_real_image(video_episode, pick_frame_ids[0])

        place_frame_ids = [
            hold_state.place_frame_id
            for hold_state in self.hold_states
        ]
        self.set_robot_states(self.episodes, place_frame_ids)
        self.arm.genesis.step()
        sim_place_image = self.get_sim_image()
        real_place_image = self.get_real_image(video_episode, place_frame_ids[0])

        return sim_pick_image, real_pick_image, sim_place_image, real_place_image

    def set_robot_states(self, episodes, frame_ids):
        robot_states = self.get_robot_states(episodes, frame_ids, EpisodeReplayer.LEADER_STATE_COLUMN)
        self.arm.genesis.entity.set_dofs_position(robot_states)

    def write_image(self, type, rgb_image, episode_id, step_id):
        image = Image.fromarray(rgb_image, mode='RGB')

        image_path = f"img/{self.repo_id}/{type}/episode_{episode_id:03d}/frame_{step_id:03d}.png"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        image.save(image_path)

    def get_real_image(self, video_episode, frame_id):
        camera_key = self.dataset.meta.camera_keys[0]
        camera_image = video_episode[frame_id][camera_key]
        camera_image = camera_image.data.numpy()
        camera_image = camera_image.transpose(1, 2, 0)

        # convert from [0-1] floats to [0-256[ ints
        camera_image = (camera_image * 255).astype("uint8")

        #self.write_image("real", camera_image, episode_id, frame_id)
        return camera_image

    def get_sim_image(self):
        rgb_image, _, _, _ = self.arm.genesis.camera.render(rgb=True)
        #self.write_image("sim", rgb_image, episode_id, frame_id)
        return rgb_image