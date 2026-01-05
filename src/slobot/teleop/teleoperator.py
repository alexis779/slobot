from slobot.feetech import Feetech
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.teleop.teleop_event import ActionEvent, SimEvent, TeleopEvent
from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame

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

        self.teleop_event = TeleopEvent(
            teleop=ActionEvent(start_time=0.0, end_time=0.0),
            leader_read=ActionEvent(start_time=0.0, end_time=0.0),
            follower_control=ActionEvent(start_time=0.0, end_time=0.0),
            follower_read=ActionEvent(start_time=0.0, end_time=0.0),
            sim_step=ActionEvent(start_time=0.0, end_time=0.0),
            simulation_frame=SimulationFrame(feetech_frame=FeetechFrame()),
        )

        self.step = 0

        # Simulation thread synchronization
        self.sim = kwargs.get('sim', False)
        if self.sim:
            from slobot.so_arm_100 import SoArm100

            self.so_arm_100 = SoArm100(mjcf_path=Configuration.MJCF_CONFIG, fps=10)

            self._sim_condition = threading.Condition()
            self._sim_event = SimEvent(step=0, control_qpos=[])
            self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._sim_thread.start()

    def _sim_loop(self):
        """Simulation thread loop - waits for new sim event and runs simulation step."""
        while True:
            with self._sim_condition:
                self._sim_condition.wait()

            self.so_arm_100.genesis.entity.control_dofs_position(self._sim_event.control_qpos)
            self.teleop_event.sim_step.start_time = time.time()
            self.so_arm_100.genesis.step()
            self.teleop_event.sim_step.end_time = time.time()
            self.teleop_event.simulation_frame.qpos = self.so_arm_100.genesis.entity.get_qpos()
            self.rerun_metrics.log_teleop_sim_event(self._sim_event.step, self.teleop_event)

    def _notify_sim(self, step: int, control_qpos: list[float]):
        """Notify the simulation thread of a new sim event and wait for completion."""
        with self._sim_condition:
            self._sim_event.step = step
            self._sim_event.control_qpos = control_qpos
            self._sim_condition.notify()

    def teleoperate(self):
        while True:
            self.teleoperate_step(self.period)

    def teleoperate_step(self, period):
        start_time = time.time()

        leader_pos = self.leader.get_pos()
        leader_qpos = self.follower.pos_to_qpos(leader_pos)
        end_leader_read_time = time.time()

        self.follower.control_position(leader_pos)
        end_follower_control_time = time.time()

        follower_pos = self.follower.get_pos()
        follower_qpos = self.follower.pos_to_qpos(follower_pos)
        end_follower_read_time = time.time()

        self.teleop_event.teleop.start_time = start_time
        self.teleop_event.leader_read.start_time = start_time
        self.teleop_event.leader_read.end_time = end_leader_read_time
        self.teleop_event.follower_control.start_time = end_leader_read_time
        self.teleop_event.follower_control.end_time = end_follower_control_time
        self.teleop_event.follower_read.start_time = end_follower_control_time
        self.teleop_event.follower_read.end_time = end_follower_read_time
        self.teleop_event.teleop.end_time = end_follower_read_time

        self.teleop_event.simulation_frame.feetech_frame.control_pos = leader_qpos
        self.teleop_event.simulation_frame.feetech_frame.qpos = follower_qpos

        self.rerun_metrics.log_teleop_real_event(self.step, self.teleop_event)

        # Notify sim thread of new sim event before sleeping
        if self.sim:
            self._notify_sim(step=self.step, control_qpos=self.teleop_event.simulation_frame.feetech_frame.control_pos)

        end_time = time.time()
        sleep_period = period - (end_time - start_time)
        if sleep_period > 0:
            time.sleep(sleep_period)

        self.step += 1
