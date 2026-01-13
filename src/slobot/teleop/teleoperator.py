from slobot.feetech import Feetech
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.teleop.teleop_event import ActionEvent, TeleopEvent
from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame

import time

class Teleoperator():
    LOGGER = Configuration.logger(__name__)

    def __init__(self, **kwargs):
        self.rerun_metrics = RerunMetrics(**kwargs)
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

        end_time = time.time()
        sleep_period = period - (end_time - start_time)
        if sleep_period > 0:
            time.sleep(sleep_period)

        self.step += 1
