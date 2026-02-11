from slobot.robotic_arm import RoboticArm
from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/0_validate_sim_qpos.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Validate the robot is located in the position preset

preset = sys.argv[1]
qpos = Configuration.QPOS_MAP[preset]
arm = SoArm100(record=True)

#RoboticArm.sim_qpos(arm, qpos)
arm.genesis.entity.control_dofs_position(qpos)

for _ in range(100):
    arm.genesis.step()

arm.genesis.stop()
