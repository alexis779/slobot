from time import sleep
from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python 5_validate_sim_to_real.py [zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset in sim then real

feetech = Feetech()

preset = sys.argv[1]
qpos = Configuration.QPOS_MAP[preset]

mjcf_path = Configuration.MJCF_CONFIG
arm = SoArm100(mjcf_path=mjcf_path, qpos_handler=feetech)
arm.genesis.entity.set_qpos(qpos)
arm.genesis.hold_entity()
