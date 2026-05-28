from slobot.teleop.gradio_joint_control_sim_app import GradioJointControlSimApp
from slobot.configuration import Configuration
from slobot.robotic_arm import RoboticArm

# SO-ARM-100
urdf_path = Configuration.URDF_CONFIG
arm = RoboticArm(urdf_path=urdf_path)

# UR5e
#mjcf_path = "../mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
#max_force = 1000
#arm = RoboticArm(mjcf_path=mjcf_path, max_force=max_force)

# ARX-X5
#urdf_path = "../ARX_Model/X5/X5A/urdf/X5A.urdf"
#max_force = 1000
#arm = RoboticArm(urdf_path=urdf_path, max_force=max_force)

# ARX-L5
#mjcf_path = "../mujoco_menagerie/arx_l5/scene.xml"
#max_force = 1000
#arm = RoboticArm(mjcf_path=mjcf_path, max_force=max_force)


gradio_app = GradioJointControlSimApp(arm)
gradio_app.launch()
