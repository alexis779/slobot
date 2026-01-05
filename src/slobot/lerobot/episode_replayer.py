from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.lerobot.episode_loader import EpisodeLoader

import torch
from dataclasses import dataclass

import genesis as gs
from genesis.engine.entities import RigidEntity

from PIL import Image

import os

from importlib.resources import files

@dataclass
class InitialState:
    ball: torch.Tensor  # 3D float tensor [x, y, z]
    cup: torch.Tensor  # 3D float tensor [x, y, z]

class EpisodeReplayer:
    LOGGER = Configuration.logger(__name__)

    FIXED_JAW_TRANSLATE = [-1.4e-2, -9e-2, 0] # the translation vector from the fixed jaw position to the ball position, in the frame relative to the link
    GOLF_BALL_RADIUS = 4.27e-2 / 2

    DISTANCE_THRESHOLD = 0.01 # the threshold for the distance between the golf ball and the cup for the ball to be considered in the cup, or for the ball to be considered moved from the initial position

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]

        self.episode_ids = kwargs["episode_ids"]
        self.episode_loader = EpisodeLoader(repo_id=self.repo_id, episode_ids=self.episode_ids)

        self.fixed_jaw_translate = torch.tensor(EpisodeReplayer.FIXED_JAW_TRANSLATE)
        # FPS
        kwargs["fps"] = self.episode_loader.dataset.meta.fps
        kwargs["should_start"] = False

        kwargs["show_viewer"] = kwargs.get("show_viewer", False)

        # Image Resolution of the 1st camera
        self.camera_key = self.episode_loader.dataset.meta.camera_keys[0]
        video_height, video_width, channels = self.episode_loader.dataset.meta.features[self.camera_key]['shape']
        self.res = (video_width, video_height)
        kwargs["res"] = self.res

        self.add_metrics = kwargs.get("add_metrics", False)
        if self.add_metrics:
            self.metrics = RerunMetrics()
            kwargs["step_handler"] = self.metrics

        self.arm = SoArm100(**kwargs)

        self.build_scene()

    def replay_episodes(self):
        moved, success = self.replay_episode_batch()

        # Log failed episodes
        failed_episode_ids = [self.episode_ids[i] for i in range(len(success)) if not success[i]]
        EpisodeReplayer.LOGGER.info(f"Failed episodes: {','.join(map(str, failed_episode_ids))}")

        score = (sum(moved) + sum(success)) / (2 * self.episode_loader.episode_count)

        return score

    def replay_episode_batch(self):
        for frame_id in range(self.episode_loader.episode_frame_count):
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

    def build_scene(self):
        self.arm.genesis.start()

        golf_ball_morph = gs.morphs.Mesh(
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
            golf_ball_morph,
            visualize_contact=False, # True
            vis_mode='visual', # collision
        )

        self.cup : RigidEntity = self.arm.genesis.scene.add_entity(cup)

        n_envs = len(self.episode_ids)
        self.arm.genesis.build(n_envs=n_envs)

        qpos_limits = self.arm.genesis.entity.get_dofs_limit()
        self.episode_loader.set_dofs_limit(qpos_limits)

        self.set_object_initial_positions()

    def replay_frame(self, frame_id):
        frame_ids = [
            frame_id
            for _ in range(len(self.episode_loader.episodes))
        ]
        leader_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.LEADER_STATE_COLUMN, frame_ids)

        #EpisodeReplayer.LOGGER.info(f"frame_id = {frame_id}")

        if frame_id == 0:
            self.arm.genesis.entity.set_dofs_position(leader_robot_states)
        else:
            self.arm.genesis.entity.control_dofs_position(leader_robot_states)

        self.arm.genesis.step()

        follower_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.FOLLOWER_STATE_COLUMN, frame_ids)
        if self.add_metrics:
            self.metrics.add_metric("real.qpos", follower_robot_states)

    def get_initial_states(self) -> list[InitialState]:
        pick_frame_ids = [
            hold_state.pick_frame_id
            for hold_state in self.episode_loader.hold_states
        ]

        self.set_robot_states(pick_frame_ids)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate)

        place_frame_ids = [
            hold_state.place_frame_id
            for hold_state in self.episode_loader.hold_states
        ]
        self.set_robot_states(place_frame_ids)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate)

        return [
            InitialState(
                ball=pick_link_pos_i,
                cup=place_link_pos_i,
            )
            for pick_link_pos_i, place_link_pos_i, in zip(pick_link_pos, place_link_pos)
        ]

    def get_initial_state_images(self):
        pick_frame_ids = [
            hold_state.pick_frame_id
            for hold_state in self.episode_loader.hold_states
        ]

        self.set_robot_states(pick_frame_ids)
        self.arm.genesis.step()
        sim_pick_image = self.get_sim_image()
        real_pick_image = self.get_real_image(pick_frame_ids[0])

        place_frame_ids = [
            hold_state.place_frame_id
            for hold_state in self.episode_loader.hold_states
        ]
        self.set_robot_states(place_frame_ids)
        self.arm.genesis.step()
        sim_place_image = self.get_sim_image()
        real_place_image = self.get_real_image(place_frame_ids[0])

        return sim_pick_image, real_pick_image, sim_place_image, real_place_image

    def set_robot_states(self, frame_ids):
        robot_states = self.episode_loader.get_robot_states(EpisodeLoader.LEADER_STATE_COLUMN, frame_ids)
        self.arm.genesis.entity.set_dofs_position(robot_states)

    def write_image(self, type, rgb_image, episode_id, step_id):
        image = Image.fromarray(rgb_image, mode='RGB')

        image_path = f"img/{self.repo_id}/{type}/episode_{episode_id:03d}/frame_{step_id:03d}.png"

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        image.save(image_path)

    def get_real_image(self, frame_id):
        camera_image = self.episode_loader.dataset[frame_id][self.camera_key]
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