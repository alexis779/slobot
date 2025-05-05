from time import sleep
import numpy as np

from slobot.feetech import Feetech
from slobot.configuration import Configuration

from lerobot.common.kinematics.kinematics import Robot, RobotKinematics, RobotUtils

# Move the robot to the 3 preset positions.

feetech = Feetech()

feetech.move(Configuration.POS_MAP['zero'])
sleep(1)
feetech.move(Configuration.POS_MAP['rotated'])
sleep(1)


# Move the end effector
kin = RobotKinematics()
robot = Robot(robot_type="so100")

pos_current = feetech.get_pos()
qpos_current = feetech.pos_to_qpos(pos_current)
qpos_current = np.array(qpos_current)

print("qpos_current", qpos_current)

qpos_current = qpos_current[:-1]


ee_pos_current = kin.forward_kinematics(robot, qpos_current)

ee_pos_goal = ee_pos_current.copy()
ee_pos_goal[:3, 3] += np.array([0.0, 0.0, +0.1])

qpos_goal = kin.inverse_kinematics(robot, qpos_current, ee_pos_goal)

qpos_goal = np.append(qpos_goal, 0)

print("qpos_goal", qpos_goal)

pos_goal = feetech.qpos_to_pos(qpos_goal)
print("pos_goal", pos_goal)

#feetech.move(pos_goal)
#sleep(10)

feetech.move(Configuration.POS_MAP['rest'])
sleep(1)
feetech.set_torque(False)