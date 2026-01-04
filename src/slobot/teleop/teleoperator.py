from slobot.feetech import Feetech
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.teleop.teleop_event import ActionEvent, TeleopEvent
from slobot.so_arm_100 import SoArm100

import threading
import time

class Teleoperator():
    LOGGER = Configuration.logger(__name__)

    def __init__(self, **kwargs):
        self.rerun_metrics = RerunMetrics()
        self.follower = Feetech(port=Feetech.PORT0, robot_id=Feetech.FOLLOWER_ID)
        self.leader = Feetech(port=Feetech.PORT1, robot_id=Feetech.LEADER_ID, torque=False)

        self.fps = kwargs['fps']
        self.period = 1/self.fps

        self.so_arm_100 = SoArm100(mjcf_path=Configuration.MJCF_CONFIG, fps=self.fps)

        self.step = 0
        self.teleop_event = TeleopEvent(
            step=0,
            teleop=ActionEvent(start_time=0.0, end_time=0.0),
            leader_read=ActionEvent(start_time=0.0, end_time=0.0),
            follower_control=ActionEvent(start_time=0.0, end_time=0.0),
            follower_read=ActionEvent(start_time=0.0, end_time=0.0),
            leader_qpos=[0.0] * Configuration.DOFS,
            follower_qpos=[0.0] * Configuration.DOFS,
        )

        # Simulation thread synchronization
        self._sim_condition = threading.Condition()
        self._sim_qpos = None
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    def _sim_loop(self):
        """Simulation thread loop - waits for new qpos and runs simulation step."""
        while True:
            with self._sim_condition:
                self._sim_condition.wait()

            self.so_arm_100.genesis.entity.control_dofs_position(self._sim_qpos)
            self.so_arm_100.genesis.step()

    def _notify_sim(self, sim_qpos):
        """Notify the simulation thread of a new follower position and wait for completion."""
        with self._sim_condition:
            self._sim_qpos = sim_qpos
            self._sim_condition.notify()

    def teleoperate(self):
        while True:
            self.teleoperate_step(self.period)

    def teleoperate_step(self, period) -> TeleopEvent:
        start_time = time.time()

        leader_pos = self.leader.get_pos()
        leader_qpos = self.follower.pos_to_qpos(leader_pos)
        end_leader_read_time = time.time()

        self.follower.control_position(leader_pos)
        end_follower_control_time = time.time()

        follower_pos = self.follower.get_pos()
        end_follower_read_time = time.time()

        # Notify sim thread of new follower position before sleeping
        follower_qpos = self.follower.pos_to_qpos(follower_pos)
        self._notify_sim(follower_qpos)

        end_time = time.time()
        sleep_period = period - (end_time - start_time)
        if sleep_period > 0:
            time.sleep(sleep_period)

        self.teleop_event.step = self.step
        self.teleop_event.teleop.start_time = start_time
        self.teleop_event.leader_read.start_time = start_time
        self.teleop_event.leader_read.end_time = end_leader_read_time
        self.teleop_event.follower_control.start_time = end_leader_read_time
        self.teleop_event.follower_control.end_time = end_follower_control_time
        self.teleop_event.follower_read.start_time = end_follower_control_time
        self.teleop_event.follower_read.end_time = end_follower_read_time
        self.teleop_event.teleop.end_time = end_time

        self.teleop_event.leader_qpos = leader_qpos
        self.teleop_event.follower_qpos = follower_qpos

        self.rerun_metrics.log_teleop_event(self.teleop_event)

        self.step += 1
        return self.teleop_event
