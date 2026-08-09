from slobot.feetech import Feetech
from slobot.robotic_arm import RoboticArm
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: uv run python scripts/validation/4_validate_real_to_sim.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset in real then sim

robotic_arm = RoboticArm(mjcf_path=Configuration.MJCF_CONFIG)

preset = sys.argv[1]
pos = Configuration.POS_MAP[preset]

feetech = Feetech(qpos_map=Configuration.MJCF_QPOS_MAP, qpos_handler=robotic_arm)
feetech.control_position(pos)

robotic_arm.genesis.hold_entity()