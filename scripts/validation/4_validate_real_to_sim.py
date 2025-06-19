from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
import sys
from time import sleep

if len(sys.argv) < 2:
    print("Usage: python 4_validate_real_to_sim.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset in real then sim

mjcf_path = Configuration.MJCF_CONFIG
arm = SoArm100(mjcf_path=mjcf_path)

preset = sys.argv[1]
pos = Configuration.POS_MAP[preset]

feetech = Feetech(qpos_handler=arm)
feetech.control_position(pos)
sleep(1)
feetech.control_position(pos)

arm.genesis.hold_entity()