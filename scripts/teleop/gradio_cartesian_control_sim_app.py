from slobot.teleop.gradio_cartesian_control_sim_app import GradioCartesianControlSimApp
from slobot.so_arm_100 import SoArm100
from slobot.robotic_arm import RoboticArm

# SO-ARM-100
#arm = SoArm100()

# ARX-X5
urdf_path = "../ARX_Model/X5/X5A/urdf/X5A.urdf"
max_force = 1000
arm = RoboticArm(urdf_path=urdf_path, max_force=max_force)

# ARX-L5
#mjcf_path = "../mujoco_menagerie/arx_l5/scene.xml"
#max_force = 1000
#arm = RoboticArm(mjcf_path=mjcf_path, max_force=max_force)

gradio_app = GradioCartesianControlSimApp(arm)
gradio_app.launch()
