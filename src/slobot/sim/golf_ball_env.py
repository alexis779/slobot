from __future__ import annotations

from typing import Callable, Optional

import genesis as gs
import numpy as np
import torch
from importlib.resources import files

from slobot.configuration import Configuration
from slobot.sim.gym.types import GolfBallEnvAction, GolfBallEnvObservation
from slobot.so_arm_100 import SoArm100


class GolfBallEnv:
    IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0])

    def __init__(self, requires_grad: bool = False, step_handler: Callable = None):
        # Create SoArm100 instance
        self.arm = SoArm100(
            should_start=False,
            rgb=True,
            vis_mode='visual', # collision
            requires_grad=requires_grad,
            step_handler=step_handler,
            show_world_frame=False, # True
        )
        
        # Build scene with objects
        self.build_scene()
        self.arm.check_dofs_limit()

        w, h = self.arm.genesis.res
        n = self.arm.genesis.entity.n_dofs
        self.observation: GolfBallEnvObservation = {
            "qpos": np.zeros(n, dtype=np.float32),
            "side_camera_image": np.zeros((h, w, 3), dtype=np.uint8),
            "link_camera_image": np.zeros((h, w, 3), dtype=np.uint8),
        }

    def build_scene(self):
        self.arm.genesis.start()

        visualize_contact = False # True

        golf_ball_morph = gs.morphs.Sphere(
            radius=Configuration.GOLF_BALL_RADIUS,
        )
        self.golf_ball = self.arm.genesis.scene.add_entity(
            golf_ball_morph,
            visualize_contact=visualize_contact,
            vis_mode=self.arm.genesis.vis_mode
        )

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup_morph = gs.morphs.Mesh(
            file=cup_filename,
        )
        self.cup = self.arm.genesis.scene.add_entity(
            cup_morph,
            visualize_contact=visualize_contact,
            vis_mode=self.arm.genesis.vis_mode
        )

        n_envs = 1
        self.arm.genesis.build(n_envs=n_envs)

    def set_object_initial_positions(self, ball_x: float, ball_y: float, cup_x: float, cup_y: float):
        ball_x = ball_x * Configuration.INCHES_TO_METERS
        ball_y = ball_y * Configuration.INCHES_TO_METERS
        cup_x = cup_x * Configuration.INCHES_TO_METERS
        cup_y = cup_y * Configuration.INCHES_TO_METERS

        self.ball_pos = torch.tensor([[
            ball_x,
            ball_y,
            Configuration.GOLF_BALL_RADIUS
        ]])
        self.cup_pos = torch.tensor([[
            cup_x,
            cup_y,
            0.0
        ]])
        self.golf_ball.set_pos(self.ball_pos)

        self.cup.set_pos(self.cup_pos)
        self.cup.set_quat(self.IDENTITY_QUAT) # sometimes the robot may tip the cup, so it needs to be straightened

        self.arm.genesis.entity.set_dofs_position(self.arm.genesis.home_qpos)

    def is_golf_ball_in_cup(self) -> bool:
        """
        Check if the golf ball is near the cup position.
        
        Returns:
            True if ball is within distance threshold of cup, False otherwise
        """
        ball_pos = self.golf_ball.get_pos()[0]
        cup_pos = self.cup.get_pos()[0]

        # Check distance in XY plane
        diff = ball_pos[:2] - cup_pos[:2]
        distance = torch.norm(diff)

        return distance < Configuration.DISTANCE_THRESHOLD

    def load_observation(self) -> None:
        q = self.arm.genesis.entity.get_qpos()[0]
        self.observation["qpos"] = q.detach().cpu().numpy().astype(np.float32)
        self.observation["side_camera_image"] = self.arm.genesis.side_camera.render()[0]
        self.observation["link_camera_image"] = self.arm.genesis.link_camera.render()[0]

    def reset(self, options: dict) -> GolfBallEnvObservation:
        recording_layout = options["recording_layout"]
        self.set_object_initial_positions(
            recording_layout.ball_x,
            recording_layout.ball_y,
            recording_layout.cup_x,
            recording_layout.cup_y,
        )
        self.sim_steps = options["sim_steps"]
        self.load_observation()
        return self.observation

    def step(self, action: GolfBallEnvAction) -> GolfBallEnvObservation:
        self.arm.genesis.entity.control_dofs_position(
            torch.as_tensor(action["control_qpos"], dtype=torch.float32)
        )
        for _ in range(self.sim_steps):
            self.arm.genesis.step()
        self.load_observation()
        return self.observation

    def close(self) -> None:
        self.arm.genesis.stop()
