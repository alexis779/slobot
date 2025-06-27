from slobot.feetech import Feetech
from slobot.metrics.metrics import Metrics
from slobot.configuration import Configuration

import time

class Teleoperator():
    def __init__(self):
        metrics = Metrics()
        self.follower = Feetech(port=Feetech.PORT0, robot_id=Feetech.FOLLOWER_ID, qpos_handler=metrics)
        self.leader = Feetech(port=Feetech.PORT1, robot_id=Feetech.LEADER_ID, torque=False)

    def teleoperate(self, fps):
        # PDI (Proportional Derivative Integral) controller gains
        print("K_p=", self.follower.get_dofs_kp())
        print("K_v=", self.follower.get_dofs_kv())
        print("K_i=", self.follower.get_dofs_ki())

        period = 1/fps

        while True:
            start_time = time.time()
            leader_pos = self.leader.get_pos()
            self.follower.control_position(leader_pos)
            end_time = time.time()

            sleep_period = period - (end_time - start_time)
            if sleep_period > 0:
                time.sleep(sleep_period)