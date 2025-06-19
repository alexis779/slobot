from slobot.so_arm_100 import SoArm100
from slobot.simulation_frame import SimulationFrame
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import torch

import genesis as gs
from genesis.engine.entities import RigidEntity

from PIL import Image

from dataclasses import dataclass

import os

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


class EpisodeReplayer:
    SCALE = torch.tensor([-1, -0.92, 1, 1, -1, 0.75]) # the scaling factor of the Follower angular position

    fixed_jaw_t = torch.tensor([-2e-2, -9e-2, 0]) # the translation vector from the fixed jaw position to the ball position, in the frame relative to the link
    golf_ball_radius = 4.27e-2 / 2

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]

        # FPS
        self.ds_meta = LeRobotDatasetMetadata(self.repo_id)
        kwargs["fps"] = self.ds_meta.fps
        kwargs["should_start"] = False
        kwargs["show_viewer"] = False

        # Image Resolution of the 1st camera
        camera_key = self.ds_meta.camera_keys[0]
        video_height, video_width, channels = self.ds_meta.features[camera_key]['shape']
        kwargs["res"] = (video_width, video_height)

        # enable RGB camera
        kwargs["step_handler"] = self
        kwargs["rgb"] = True

        self.arm = SoArm100(**kwargs)

    def replay_episodes(self):
        episode_count = self.ds_meta.total_episodes

        success = 0
        for episode_id in range(episode_count):
            success += 1 if self.replay_episode(episode_id) else 0

        return success / episode_count

    def replay_episode(self, episode_id):
        self.step_id = 0
        self.episode_id = episode_id

        dataset = LeRobotDataset(self.repo_id, episodes=[episode_id])

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()
        episode_frame_count = to_idx - from_idx

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=episode_frame_count,
        )

        episode = next(iter(dataloader))

        hold_state = self.get_hold_state(episode)

        self.add_objects(episode, hold_state)

        for frame_id in range(episode_frame_count):
            self.replay_frame(episode, frame_id, hold_state)

        golf_ball_pos = self.golf_ball.get_pos()
        cup_pos = self.cup.get_pos()

        self.arm.genesis.stop()

        distance = torch.dist(golf_ball_pos[:2], cup_pos[:2]) # project error in the XY plane

        distance_threshold = 0.01
        return distance < distance_threshold

    def write_episodes_images(self):
        episode_count = self.ds_meta.total_episodes
        for episode_id in range(episode_count):
            self.write_episode_images(episode_id)

    def write_episode_images(self, episode_id):
        dataset = LeRobotDataset(self.repo_id, episodes=[episode_id])

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()
        episode_frame_count = to_idx - from_idx

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=episode_frame_count,
        )

        episode = next(iter(dataloader))
        for frame_id in range(episode_frame_count):
            self.write_camera_image(episode, frame_id)

    def add_objects(self, episode, hold_state : HoldState):
        self.arm.genesis.kwargs["show_viewer"] = False # disable visualization
        os.environ['PYOPENGL_PLATFORM'] = 'egl' # offline mode

        self.arm.genesis.start()
        self.arm.genesis.build()
        self.qpos_limits = self.arm.genesis.entity.get_dofs_limit()

        # compute the initial positions of the ball and the cup
        initial_state : InitialState = self.get_initial_state(episode, hold_state)

        self.arm.genesis.stop()

        self.arm.genesis.kwargs["show_viewer"] = True # False
        self.arm.genesis.start()

        golf_ball = gs.morphs.Sphere(
            radius=self.golf_ball_radius,
            pos=(initial_state.ball_x, initial_state.ball_y, self.golf_ball_radius),
        )

        cup = gs.morphs.Mesh(
            file="./doc/cup.stl",
            pos=(initial_state.cup_x, initial_state.cup_y, 0),
        )

        self.golf_ball : RigidEntity = self.arm.genesis.scene.add_entity(
            golf_ball,
            visualize_contact=False # True
        )
        self.cup : RigidEntity = self.arm.genesis.scene.add_entity(cup)

        os.environ['PYOPENGL_PLATFORM'] = 'glx' # egl
        self.arm.genesis.build()

    def replay_frame(self, episode, frame_id, hold_state : HoldState):
        robot_state = self.get_robot_state(episode, frame_id)
        if frame_id == 0:
            self.arm.genesis.entity.set_qpos(robot_state)
        else:
            self.arm.genesis.entity.control_dofs_position(robot_state)

        if frame_id == hold_state.pick_frame_id or frame_id == hold_state.place_frame_id:
            color = (0, 1, 1, 0.4)
            #self.arm.genesis.draw_arrow(self.arm.genesis.fixed_jaw, self.fixed_jaw_t, color)

        self.write_camera_image(episode, frame_id)

        self.arm.genesis.step()
    
    def get_robot_state(self, episode, frame_id):
        robot_state = episode['observation.state'][frame_id]
        return self.degrees_to_radians(robot_state)

    def degrees_to_radians(self, degrees):
        radians = torch.deg2rad(degrees)
        radians = radians * self.SCALE.to(radians.device)
        radians = torch.clamp(radians, self.qpos_limits[0], self.qpos_limits[1])
        return radians

    def get_hold_state(self, episode) -> HoldState:
        gripper_id = 5 # the id of the jaw joint
        follower_gripper = episode['action'][:,gripper_id].cpu()
        leader_gripper = episode['observation.state'][:,gripper_id].cpu()

        delay_frames = 3 # the number of fps the follower takes to reflect the leader position
        truncated_leader = leader_gripper[delay_frames:]
        gripper_diff = truncated_leader - follower_gripper[:-delay_frames]

        diff_threshold = 15 # the cutoff value to identify when the gripper is holding the ball
        above_threshold = torch.where(gripper_diff > diff_threshold, 1, 0)
        above_threshold_derivative = torch.diff(above_threshold, prepend=above_threshold[0:1])

        pick_frame_id, = torch.where(above_threshold_derivative == 1)
        place_frame_id, = torch.where(above_threshold_derivative == -1)
        pick_frame_id = pick_frame_id.item()
        place_frame_id = place_frame_id.item()

        return HoldState(pick_frame_id=pick_frame_id, place_frame_id=place_frame_id)

    def get_initial_state(self, episode, hold_state: HoldState):
        self.set_robot_state(episode, hold_state.pick_frame_id)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_t)

        self.set_robot_state(episode, hold_state.place_frame_id)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_t)

        return InitialState(ball_x=pick_link_pos[0].item(), ball_y=pick_link_pos[1].item(), cup_x=place_link_pos[0].item(), cup_y=place_link_pos[1].item())

    def set_robot_state(self, episode, frame_id):
        robot_state = self.get_robot_state(episode, frame_id)
        self.arm.genesis.entity.set_qpos(robot_state)

    def handle_step(self, simulation_frame: SimulationFrame):
        self.write_image("sim", simulation_frame.rgb, self.episode_id, self.step_id)
        self.step_id += 1

    def write_image(self, type, rgb_image, episode_id, step_id):
        image = Image.fromarray(rgb_image, mode='RGB')

        image_path = f"img/{type}/episode_{episode_id:03d}/frame_{step_id:03d}.png"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        image.save(image_path)

    def write_camera_image(self, episode, frame_id):
        camera_key = self.ds_meta.camera_keys[0]
        camera_image = episode[camera_key][frame_id]
        camera_image = camera_image.data.numpy()
        camera_image = camera_image.transpose(1, 2, 0)

        # convert from [0-1] floats to [0-256[ ints
        camera_image = (camera_image * 255).astype("uint8")

        episode_id = episode['episode_index'][frame_id].item()
        self.write_image("real", camera_image, episode_id, frame_id)
