from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100
from slobot.robotic_arm import RoboticArm
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/2_validate_sim_pos.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset

feetech = Feetech()

preset = sys.argv[1]
pos = Configuration.POS_MAP[preset]
qpos = feetech.pos_to_qpos(pos)

arm = SoArm100()
RoboticArm.sim_qpos(arm, qpos)