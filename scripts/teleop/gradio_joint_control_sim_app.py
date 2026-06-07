from slobot.teleop.gradio_joint_control_sim_app import GradioJointControlSimApp
from slobot.robotic_arm import RoboticArm

# SO-ARM-100
urdf_path = "../SO-ARM100/Simulation/SO100/so100.urdf"
#urdf_path = "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf" # SO-ARM-101
robotic_arm = RoboticArm(urdf_path=urdf_path)

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


gradio_app = GradioJointControlSimApp(robotic_arm)
gradio_app.launch()
