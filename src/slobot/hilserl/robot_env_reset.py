"""Shared RobotEnv reset: pause for manual scene setup, then move the arm home."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from slobot.hilserl.handlers.motor_qpos import GRIPPER_MOTOR_IDX

if TYPE_CHECKING:
    from slobot.hilserl.slobot_so100_follower import SlobotSO100Follower
    from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop

def _log_reset_stage(message: str) -> None:
    logging.info(message)
    print(message, flush=True)


def sleep_with_gui_pump(seconds: float, leader_teleop: SlobotSO100LeaderTeleop) -> None:
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        leader_teleop.pump_gui()
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        precise_sleep(min(0.05, remaining))


def wait_control_step(step_start: float, fps: float, leader_teleop: SlobotSO100LeaderTeleop) -> None:
    """Sleep until one control period (1/fps) has elapsed since step_start."""
    remaining = max(1.0 / fps - (time.perf_counter() - step_start), 0.0)
    sleep_with_gui_pump(remaining, leader_teleop)


def reset_jaw_to_pose(
    robot: SlobotSO100Follower,
    target_radians: list[float] | np.ndarray,
) -> None:
    """Command only the jaw to the reset pose, leaving arm joints at their present qpos."""
    target_qpos = np.asarray(target_radians, dtype=np.float32).tolist()
    qpos = list(robot._feetech.get_qpos())
    qpos[GRIPPER_MOTOR_IDX] = target_qpos[GRIPPER_MOTOR_IDX]
    robot._feetech.control_dofs_position(qpos)


def reset_follower_to_joint_radians(
    robot: SlobotSO100Follower,
    target_radians: list[float] | np.ndarray,
) -> None:
    """Send one goal-position command to move the follower to ``target_radians``."""
    target_qpos = np.asarray(target_radians, dtype=np.float32).tolist()
    robot._feetech.control_dofs_position(target_qpos)


def robot_env_reset(
    env,
    leader_teleop: SlobotSO100LeaderTeleop,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict]:
    """Wait ``reset_time_s`` for manual scene reset, then move follower to ``reset_pose`` (radians)."""
    reset_started_at = time.perf_counter()
    reset_started_ts = datetime.now().isoformat(timespec="milliseconds")
    _log_reset_stage(
        f"Reset stage started at {reset_started_ts} (reset_time_s={env.reset_time_s:.1f}s)"
    )
    log_say("Reset the environment.", play_sounds=True)

    leader_teleop.notify_resetting()
    leader_teleop.reset_gui_for_episode()
    if env.reset_pose is not None:
        # Open the jaw before the manual wait so episode 1 has time to reach the
        # reset pose (full arm homing still runs after the wait).
        reset_jaw_to_pose(env.robot, env.reset_pose)
    manual_wait_started_at = time.perf_counter()
    sleep_with_gui_pump(env.reset_time_s, leader_teleop)
    manual_wait_elapsed_s = time.perf_counter() - manual_wait_started_at

    arm_homing_elapsed_s = 0.0
    if env.reset_pose is not None:
        arm_homing_started_at = time.perf_counter()
        reset_follower_to_joint_radians(env.robot, env.reset_pose)
        arm_homing_elapsed_s = time.perf_counter() - arm_homing_started_at

    reset_elapsed_s = time.perf_counter() - reset_started_at
    reset_ended_ts = datetime.now().isoformat(timespec="milliseconds")
    _log_reset_stage(
        f"Reset stage ended at {reset_ended_ts} "
        f"(elapsed={reset_elapsed_s:.2f}s, manual_wait={manual_wait_elapsed_s:.2f}s, "
        f"arm_homing={arm_homing_elapsed_s:.2f}s)"
    )
    log_say("Reset the environment done.", play_sounds=True)

    gym.Env.reset(env, seed=seed, options=options)
    env.current_step = 0
    env.episode_data = None
    obs = env._get_observation()
    env._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in env._joint_names}
    return obs, {TeleopEvents.IS_INTERVENTION: False}
