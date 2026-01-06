from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/4_validate_real_to_sim.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset in real then sim

mjcf_path = Configuration.MJCF_CONFIG
so_arm_100 = SoArm100(mjcf_path=mjcf_path)

preset = sys.argv[1]
pos = Configuration.POS_MAP[preset]

feetech = Feetech(qpos_handler=so_arm_100)
feetech.control_position(pos)

so_arm_100.genesis.hold_entity()