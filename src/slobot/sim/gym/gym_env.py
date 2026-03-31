from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, SupportsFloat

from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.sim.gym.types import GolfBallEnvAction, GolfBallEnvObservation


class GymEnv(gym.Env[GolfBallEnvObservation, GolfBallEnvAction]):
    def __init__(self, action_space_seed: int = 3):
        super().__init__()
        self.golf_ball_env = GolfBallEnv()
        entity = self.golf_ball_env.arm.genesis.entity
        lower, upper = entity.get_dofs_limit()
        control_box = spaces.Box(
            low=lower.numpy(),
            high=upper.numpy(),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {"control_qpos": control_box}, seed=action_space_seed
        )
        w, h = self.golf_ball_env.arm.genesis.res
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=lower.numpy(),
                    high=upper.numpy(),
                    dtype="float32",
                ),
                "side_camera_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, 3),
                    dtype=np.uint8,
                ),
                "link_camera_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[GolfBallEnvObservation, dict]:
        super().reset(seed=seed)
        obs = self.golf_ball_env.reset(options)
        return obs, {}

    def step(
        self, action: GolfBallEnvAction
    ) -> tuple[GolfBallEnvObservation, SupportsFloat, bool, bool, dict]:
        obs = self.golf_ball_env.step(action)
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        return self.golf_ball_env.close()