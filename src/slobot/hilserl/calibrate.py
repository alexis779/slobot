"""Run LeRobot calibrate with slobot SO-100 follower and leader plugins."""

from __future__ import annotations

# Register custom robot and teleop configs before LeRobot parses CLI args.
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401

from lerobot.scripts.lerobot_calibrate import main
