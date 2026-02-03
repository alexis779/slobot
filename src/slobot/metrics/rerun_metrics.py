from rerun.datatypes.color_model import ColorModel
from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame
from slobot.configuration import Configuration

import rerun as rr
import os
import av

from enum import Enum

class OperationMode(Enum):
    SAVE = "save"
    SPAWN = "spawn"
    GRPC = "grpc"

class RerunMetrics:
    LOGGER = Configuration.logger(__name__)
    APPLICATION_ID = "teleoperation"
    RRD_FOLDER = f"{Configuration.WORK_DIR}/{APPLICATION_ID}"
    TIME_METRIC = "step"
    CONTROL_POS_METRIC = "/leader/qpos"
    REAL_QPOS_METRIC = "/follower/qpos"

    def __init__(self, **kwargs):
        self.operation_mode = kwargs['operation_mode']
        self.worker_name = kwargs.get('worker_name', 'worker')

        self.step = 0

    def init_rerun(self, recording_id: str):
        RerunMetrics.LOGGER.info(f"Initializing recording ID {recording_id} for application {RerunMetrics.APPLICATION_ID} and worker {self.worker_name}")

        rr.init(RerunMetrics.APPLICATION_ID, recording_id=recording_id)

        match self.operation_mode:
            case OperationMode.SAVE:
                rrd_folder = f"{Configuration.WORK_DIR}/{recording_id}"
                os.makedirs(rrd_folder, exist_ok=True)
                rrd_file = f"{rrd_folder}/{self.worker_name}.rrd"
                rr.save(rrd_file)
                RerunMetrics.LOGGER.info("Recording %s started.", rrd_file)
            case OperationMode.SPAWN:
                rr.spawn()
            case OperationMode.GRPC:
                rr.connect_grpc()

        self.add_joint_metric_labels()

    def add_video_stream(self, metric_name: str):
        rr.log(metric_name, rr.VideoStream(codec="h264"), static=True)

    def add_joint_metric_labels(self):
        self.add_child_metric_label(f"/latency", self.worker_name, f"{self.worker_name} latency (ms)")

        for joint_name in Configuration.JOINT_NAMES:
            self.add_child_metric_label(RerunMetrics.CONTROL_POS_METRIC, joint_name, f"Leader {joint_name}")
            self.add_child_metric_label(RerunMetrics.REAL_QPOS_METRIC, joint_name, f"Real Follower {joint_name}")
            self.add_child_metric_label("/follower/velocity", joint_name, f"Real Velocity {joint_name}")
            self.add_child_metric_label("/follower/control_force", joint_name, f"Real Control Force {joint_name}")
            self.add_child_metric_label("/sim/qpos", joint_name, f"Sim Follower {joint_name}")
            self.add_child_metric_label("/sim/velocity", joint_name, f"Sim Velocity {joint_name}")
            self.add_child_metric_label("/sim/control_force", joint_name, f"Sim Control Force {joint_name}")

    def handle_qpos(self, feetech_frame: FeetechFrame):
        RerunMetrics.LOGGER.debug(f"Feetech frame {feetech_frame}")

        self.set_time(self.step)
        self.log_real_qpos(feetech_frame)
        self.step += 1

    def handle_step(self, simulation_frame: SimulationFrame):
        RerunMetrics.LOGGER.debug(f"Simulation frame {simulation_frame}")

        self.set_time(self.step)
        self.log_sim_qpos(simulation_frame)
        if simulation_frame.feetech_frame is not None:
            self.log_real_qpos(simulation_frame.feetech_frame)

        self.step += 1

    def log_sim_qpos(self, simulation_frame: SimulationFrame):
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric("/sim/qpos", joint_name, simulation_frame.qpos[0][i])
            if simulation_frame.control_pos is not None:
                self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, simulation_frame.control_pos[0][i])
            if simulation_frame.velocity is not None:
                self.add_metric("/sim/velocity", joint_name, simulation_frame.velocity[0][i])
            if simulation_frame.control_force is not None:
                self.add_metric("/sim/control_force", joint_name, simulation_frame.control_force[0][i])

    def log_real_qpos(self, feetech_frame: FeetechFrame):
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, feetech_frame.control_pos[i])
            self.add_metric(RerunMetrics.REAL_QPOS_METRIC, joint_name, feetech_frame.qpos[i])
            if feetech_frame.velocity is not None:
                self.add_metric("/follower/velocity", joint_name, feetech_frame.velocity[i])
            if feetech_frame.control_force is not None:
                self.add_metric("/follower/control_force", joint_name, feetech_frame.control_force[i])

    def add_metric(self, metric_name, joint_name, metric_value):
        rr.log(f"{metric_name}/{joint_name}", rr.Scalars(metric_value))

    def add_child_metric_label(self, prefix_name, child_name, label):
        self.add_metric_label(f"{prefix_name}/{child_name}", label)

    def add_metric_label(self, metric_name, label):
        rr.log(metric_name, rr.SeriesLines(names=label), static=True)

    def set_time(self, step: int):
        rr.set_time(RerunMetrics.TIME_METRIC, sequence=step)

    def log_latency(self, step: int, worker_name: str, latency_ms: float):
        self.set_time(step)
        rr.log(f"/latency/{worker_name}", rr.Scalars(latency_ms))

    def log_qpos(self, step: int, worker_name: str, qpos: list[float]):
        self.set_time(step)
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(f"/{worker_name}/qpos", joint_name, qpos[i])

    def log_control_force(self, step: int, worker_name: str, control_force: list[float]):
        self.set_time(step)
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(f"/{worker_name}/control_force", joint_name, control_force[i])

    def log_frame(self, step: int, video_metric: str, frame: av.VideoFrame, stream: av.VideoStream):
        self.set_time(step)
        for packet in stream.encode(frame):
            if packet.pts is None:
                continue

            rr.log(video_metric, rr.VideoStream.from_fields(sample=bytes(packet)))

    def log_boxes2D(self, step: int, metric_name: str, boxes, labels):
        self.set_time(step)
        rr.log(
            metric_name,
            rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
            ),
        )

    def log_points2D(self, step: int, metric_name: str, points):
        self.set_time(step)
        rr.log(
            metric_name,
            rr.Points2D(points, radii=3),
        )

    def log_raw_frame(self, step: int, metric_name: str, frame):
        """Log a raw bitmap frame to rerun.io.

        Args:
            step: The step/frame number
            metric_name: The metric name/path to log to
            frame: Raw RGB frame as numpy array or similar (H x W x 3)
        """
        self.set_time(step)
        rr.log(metric_name, rr.Image(frame))