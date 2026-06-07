"""Run HIL-SERL actor with leader teleop and gripper_link EE pose/actions."""

from __future__ import annotations

from slobot.hilserl.gui_init import init_x11_threading

init_x11_threading()

# Register custom robot, teleop, and processor configs before LeRobot imports.
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config

register_hilserl_processor_config()

import gymnasium as gym
from lerobot.processor import TransitionKey
from lerobot.rl import gym_manipulator
from lerobot.rl import actor as lerobot_actor
from lerobot.rl.actor import actor_cli

from slobot.hilserl.actor_loop import install_actor_recover_loop, install_paced_act_with_policy
from slobot.hilserl.gui_init import make_robot_env_with_genesis_before_tk
from slobot.hilserl.hilserl_gui_state import HilSerlGuiContext
from slobot.hilserl.hilserl_processors import make_hilserl_processors
from slobot.hilserl.factory import Factory
from slobot.hilserl.logging_utils import patch_init_logging_for_tracebacks
from slobot.hilserl.robot_env_reset import robot_env_reset
from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop

Factory.install()

_orig_make_robot_env = gym_manipulator.make_robot_env
_orig_reset_and_build_transition = gym_manipulator.reset_and_build_transition
_leader_teleop: SlobotSO100LeaderTeleop | None = None


def make_robot_env(cfg):
    global _leader_teleop
    if cfg.name == "gym_hil":
        env, teleop_device = _orig_make_robot_env(cfg)
    else:
        env, teleop_device = make_robot_env_with_genesis_before_tk(cfg, factory=Factory)
    if not isinstance(teleop_device, SlobotSO100LeaderTeleop):
        raise TypeError(
            f"HIL-SERL actor requires slobot_so100_leader teleop, got {type(teleop_device).__name__}"
        )
    _leader_teleop = teleop_device
    teleop_device.set_control_fps(cfg.fps)
    teleop_device.set_gui_context(HilSerlGuiContext.ACTOR)
    env.leader_teleop = teleop_device
    if cfg.name != "gym_hil" and cfg.processor.reset is not None:
        env.reset_time_s = cfg.processor.reset.reset_time_s
    return env, teleop_device


def patched_robot_env_reset(self, *, seed=None, options=None):
    if _leader_teleop is None:
        raise RuntimeError("Leader teleop not initialized; call make_robot_env first")
    return robot_env_reset(self, _leader_teleop, seed=seed, options=options)


def reset_and_build_transition(env, env_processor, action_processor):
    """Reset and return a transition that cannot immediately end the next episode."""
    transition = _orig_reset_and_build_transition(env, env_processor, action_processor)
    transition[TransitionKey.DONE] = False
    transition[TransitionKey.TRUNCATED] = False
    return transition


gym_manipulator.make_processors = make_hilserl_processors
gym_manipulator.make_robot_env = make_robot_env
gym_manipulator.RobotEnv.reset = patched_robot_env_reset
gym_manipulator.reset_and_build_transition = reset_and_build_transition

lerobot_actor.make_processors = make_hilserl_processors
lerobot_actor.make_robot_env = make_robot_env
lerobot_actor.reset_and_build_transition = reset_and_build_transition
install_paced_act_with_policy(lambda: _leader_teleop)
install_actor_recover_loop(lambda: _leader_teleop)
patch_init_logging_for_tracebacks(lerobot_actor)
