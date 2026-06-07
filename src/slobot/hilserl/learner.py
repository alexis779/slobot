"""Run HIL-SERL learner with slobot plugins (custom robot and teleop configs)."""

from __future__ import annotations

# Register custom robot, teleop, and processor configs before LeRobot parses CLI args.
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config

register_hilserl_processor_config()

from lerobot.rl.learner import train_cli
