from time import sleep
from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/5_validate_sim_to_real.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset in sim then real

feetech = Feetech()

preset = sys.argv[1]
qpos = Configuration.QPOS_MAP[preset]

arm = SoArm100(step_handler=feetech)
arm.genesis.entity.control_dofs_position(qpos)
arm.genesis.hold_entity()
