from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame
from slobot.configuration import Configuration
from slobot.teleop.teleop_event import TeleopEvent

import rerun as rr
import uuid
import os

class RerunMetrics:
    LOGGER = Configuration.logger(__name__)
    APPLICATION_ID = "teleoperation"
    RRD_FOLDER = f"{Configuration.WORK_DIR}/{APPLICATION_ID}"
    TIME_METRIC = "step"
    CONTROL_POS_METRIC = "/control_pos"
    REAL_QPOS_METRIC = "/real/qpos"

    def __init__(self):
        rrd_id = str(uuid.uuid4())
        rr.init(RerunMetrics.APPLICATION_ID, recording_id=rrd_id)

        os.makedirs(RerunMetrics.RRD_FOLDER, exist_ok=True)
        rrd_file = f"{RerunMetrics.RRD_FOLDER}/{rrd_id}.rrd"
        rr.save(rrd_file)

        for joint_name in Configuration.JOINT_NAMES:
            self.add_joint_metric_label(RerunMetrics.REAL_QPOS_METRIC, joint_name, f"Real Follower {joint_name}")
            self.add_joint_metric_label(RerunMetrics.CONTROL_POS_METRIC, joint_name, f"Leader {joint_name}")
            self.add_joint_metric_label("/real/velocity", joint_name, f"Real Velocity {joint_name}")
            self.add_joint_metric_label("/real/control_force", joint_name, f"Real Control Force {joint_name}")
            self.add_joint_metric_label("/sim/qpos", joint_name, f"Sim Follower {joint_name}")
            self.add_joint_metric_label("/sim/velocity", joint_name, f"Sim Velocity {joint_name}")
            self.add_joint_metric_label("/sim/control_force", joint_name, f"Sim Control Force {joint_name}")

        self.add_metric_label("/teleop/teleop_start", "Teleop Start")
        self.add_metric_label("/teleop/teleop_duration", "Teleop Duration")
        self.add_metric_label("/teleop/leader_read_start", "Leader Read Start")
        self.add_metric_label("/teleop/leader_read_duration", "Leader Read Duration")
        self.add_metric_label("/teleop/follower_control_start", "Follower Control Start")
        self.add_metric_label("/teleop/follower_control_duration", "Follower Control Duration")
        self.add_metric_label("/teleop/follower_read_start", "Follower Read Start")
        self.add_metric_label("/teleop/follower_read_duration", "Follower Read Duration")

        self.step = 0
        RerunMetrics.LOGGER.info("Recording %s started", rrd_file)

    def handle_qpos(self, feetech_frame: FeetechFrame):
        RerunMetrics.LOGGER.debug(f"Feetech frame {feetech_frame}")

        rr.set_time(RerunMetrics.TIME_METRIC, sequence=self.step)
        self.log_qpos(feetech_frame)
        self.step += 1

    def handle_step(self, simulation_frame: SimulationFrame):
        RerunMetrics.LOGGER.debug(f"Simulation frame {simulation_frame}")

        rr.set_time(RerunMetrics.TIME_METRIC, sequence=self.step)

        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric("/sim/qpos", joint_name, simulation_frame.qpos[0][i])
            if simulation_frame.control_pos is not None:
                self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, simulation_frame.control_pos[0][i])
            self.add_metric("/sim/velocity", joint_name, simulation_frame.velocity[0][i])
            self.add_metric("/sim/control_force", joint_name, simulation_frame.control_force[0][i])

        if simulation_frame.feetech_frame is not None:
            self.log_qpos(simulation_frame.feetech_frame)

        self.step += 1

    def log_qpos(self, feetech_frame: FeetechFrame):
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, feetech_frame.control_pos[i])
            self.add_metric(RerunMetrics.REAL_QPOS_METRIC, joint_name, feetech_frame.qpos[i])
            if feetech_frame.velocity is not None:
                self.add_metric("/real/velocity", joint_name, feetech_frame.velocity[i])
            if feetech_frame.control_force is not None:
                self.add_metric("/real/control_force", joint_name, feetech_frame.control_force[i])

    def add_metric(self, metric_name, joint_name, metric_value):
        rr.log(f"{metric_name}/{joint_name}", rr.Scalars(metric_value))

    def add_joint_metric_label(self, metric_name, joint_name, label):
        self.add_metric_label(f"{metric_name}/{joint_name}", label)

    def add_metric_label(self, metric_name, label):
        rr.log(metric_name, rr.SeriesLines(names=label), static=True)

    def log_teleop_event(self, teleop_event: TeleopEvent):
        RerunMetrics.LOGGER.debug(f"Teleop event {teleop_event}")

        rr.set_time(RerunMetrics.TIME_METRIC, sequence=teleop_event.step)
        rr.log("/teleop/teleop_start", rr.Scalars(teleop_event.teleop.start_time))
        rr.log("/teleop/teleop_duration", rr.Scalars(teleop_event.teleop.duration))
        rr.log("/teleop/leader_read_start", rr.Scalars(teleop_event.leader_read.start_time))
        rr.log("/teleop/leader_read_duration", rr.Scalars(teleop_event.leader_read.duration))
        rr.log("/teleop/follower_control_start", rr.Scalars(teleop_event.follower_control.start_time))
        rr.log("/teleop/follower_control_duration", rr.Scalars(teleop_event.follower_control.duration))
        rr.log("/teleop/follower_read_start", rr.Scalars(teleop_event.follower_read.start_time))
        rr.log("/teleop/follower_read_duration", rr.Scalars(teleop_event.follower_read.duration))

        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, teleop_event.leader_qpos[i])
            self.add_metric(RerunMetrics.REAL_QPOS_METRIC, joint_name, teleop_event.follower_qpos[i])