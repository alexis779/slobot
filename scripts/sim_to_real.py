from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration

# Control the robot via Inverse Kinematics against 3 elemental rotations. Each rotation is done in 2 steps.

feetech = Feetech()

arm = SoArm100(step_handler=feetech, show_viewer=True, rgb=True)
 arm.elemental_rotations()
feetech.go_to_rest()
arm.genesis.stop()