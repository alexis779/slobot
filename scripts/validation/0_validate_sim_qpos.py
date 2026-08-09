from slobot.robotic_arm import RoboticArm
from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: uv run python scripts/validation/0_validate_sim_qpos.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset

preset = sys.argv[1]
arm = SoArm100()
qpos = arm.preset_qpos(preset)

RoboticArm.sim_qpos(arm, qpos)