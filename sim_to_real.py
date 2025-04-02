from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech
from slobot.configuration import Configuration

# Control the robot via Inverse Kinematics against 3 elemental rotations. Each rotation is done in 2 steps.

feetech = Feetech()

mjcf_path = Configuration.MJCF_CONFIG
arm = SoArm100(mjcf_path=mjcf_path, step_handler=feetech, show_viewer=True)
arm.elemental_rotations()

feetech.go_to_rest()