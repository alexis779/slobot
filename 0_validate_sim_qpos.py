from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python 0_validate_sim_qpos.py [zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset

preset = sys.argv[1]
qpos = Configuration.QPOS_MAP[preset]
SoArm100.sim_qpos(qpos)