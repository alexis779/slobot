import numpy as np

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100

from lerobot.common.kinematics.kinematics import Robot, RobotKinematics

# Move the end effector
kin = RobotKinematics()
robot = Robot(robot_type="so100")

qpos_current0 = Configuration.QPOS_MAP['rotated']
qpos_current = np.array(qpos_current0)

print("qpos_current", qpos_current)

qpos_current_dh = robot.from_mech_to_dh(qpos_current)

world_T_tool_current = kin.forward_kinematics(robot, qpos_current_dh)

world_T_tool_goal = world_T_tool_current.copy()

fixed_jaw_id = 3

dz_goal = -0.05

world_T_tool_goal[:3, fixed_jaw_id] += np.array([0.0, 0.0, dz_goal])

qpos_goal_dh = kin.inverse_kinematics(robot, qpos_current_dh, world_T_tool_goal, use_orientation=True)

world_T_tool_actual = kin.forward_kinematics(robot, qpos_goal_dh)

dz = world_T_tool_actual[2, fixed_jaw_id] - world_T_tool_current[2, fixed_jaw_id]

print(f"Translation of z={dz}")

qpos_goal = robot.from_dh_to_mech(qpos_goal_dh)

# Append mobile jaw variable
qpos_goal0 = np.append(qpos_goal, qpos_current[Configuration.DOFS-1])

print("qpos_goal", qpos_goal0)

SoArm100.sim_qpos(qpos_goal0)