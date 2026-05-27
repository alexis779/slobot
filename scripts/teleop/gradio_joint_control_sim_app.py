from slobot.teleop.gradio_joint_control_sim_app import GradioJointControlSimApp
from slobot.so_arm_100 import SoArm100
from slobot.robotic_arm import RoboticArm

# SO-ARM-100
#urdf_path = SoArm100.SO_ARM_100_URDF_CONFIG
#max_force = 14
#arm = RoboticArm(urdf_path=urdf_path, max_force=max_force)

# UR5e
#mjcf_path = "../mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
#max_force = 1000
#arm = RoboticArm(mjcf_path=mjcf_path, max_force=max_force)

# ARX-5
urdf_path = "../ARX_Model/X5/X5A/urdf/X5A.urdf"
max_force = 100
arm = RoboticArm(urdf_path=urdf_path, max_force=max_force)

gradio_app = GradioJointControlSimApp(arm)
gradio_app.launch()
