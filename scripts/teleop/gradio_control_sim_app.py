from slobot.teleop.gradio_control_sim_app import GradioControlSimApp
from slobot.so_arm_100 import SoArm100
from slobot.robotic_arm import RoboticArm

arm = SoArm100()
#mjcf_path = "../mujoco_menagerie/franka_emika_panda/panda.xml"
#arm = RoboticArm(mjcf_path=mjcf_path, max_force=100)

gradio_app = GradioControlSimApp(arm)
gradio_app.launch()