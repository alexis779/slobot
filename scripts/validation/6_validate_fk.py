from slobot.configuration import Configuration
from slobot.robot_link_frame_plot import RobotLinkFramePlot
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/6_validate_fk.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the forward kinematics of the robot

preset = sys.argv[1]

qpos = Configuration.URDF_QPOS_MAP[preset]
qpos = np.array(qpos)

robot_link_frame_plot = RobotLinkFramePlot(Configuration.URDF_CONFIG, Configuration.URDF_GRIPPER_LINK_NAME)
robot_link_frame_plot.plot_link_frames(qpos)