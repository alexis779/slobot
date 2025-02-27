from time import sleep

from slobot.feetech import Feetech
from slobot.configuration import Configuration

# Move the robot to the 3 preset positions.

feetech = Feetech()

feetech.move(Configuration.POS_MAP['zero'])
sleep(1)
feetech.move(Configuration.POS_MAP['rotated'])
sleep(1)
feetech.move(Configuration.POS_MAP['rest'])
sleep(1)
feetech.set_torque(False)