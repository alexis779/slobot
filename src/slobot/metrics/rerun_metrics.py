from rerun.datatypes.color_model import ColorModel
from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame, CameraFrame
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

    TIME_METRIC = "step"

    CONTROL_POS_METRIC = "/leader/qpos"
    REAL_QPOS_METRIC = "/follower/qpos"

    SIM_SIDE_VIDEO_METRIC = "/sim/side/video"
    SIM_LINK_VIDEO_METRIC = "/sim/link/video"

    def __init__(self, **kwargs):
        self.operation_mode = kwargs['operation_mode']
        self.worker_name = kwargs.get('worker_name', 'worker')

        self.steps = []  # sequence of step ids that had a frame added to the video stream

        self.container = av.open("/dev/null", "w", format="h264")
        self.streams = {}

    def recording_path(self, recording_id: str) -> str:
        return f"{Configuration.WORK_DIR}/recordings/{recording_id}.rrd"

    def init_rerun(self, recording_id: str):
        self.LOGGER.info(f"Initializing recording ID {recording_id} for application {RerunMetrics.APPLICATION_ID} and worker {self.worker_name}")

        rr.init(RerunMetrics.APPLICATION_ID, recording_id=recording_id)

        match self.operation_mode:
            case OperationMode.SAVE:
                rrd_folder = f"{Configuration.WORK_DIR}/recordings"
                os.makedirs(rrd_folder, exist_ok=True)
                rrd_file = self.recording_path(recording_id)
                rr.save(rrd_file)
                self.LOGGER.info("Recording %s started.", rrd_file)
            case OperationMode.SPAWN:
                rr.spawn()
            case OperationMode.GRPC:
                rr.connect_grpc()

        self.add_joint_metric_labels()

        self.step = 0
        self.steps.clear()

    def end_recording(self):
        """Flush and close the current recording stream so the RRD file has a valid footer."""
        if self.operation_mode == OperationMode.SAVE:
            rr.disconnect()

    def create_stream(self, fps: int):
        stream = self.container.add_stream("libx264", rate=fps)
        stream.max_b_frames = 0 # current limitation of rerun.io
        return stream

    def flush_streams(self):
        for metric_name in self.streams:
            self.encode_frame(metric_name, None)

    def close_container(self):
        self.flush_streams()

        # Close the container
        self.container.close()

    def add_video_stream(self, metric_name: str, fps: int):
        rr.log(metric_name, rr.VideoStream(codec="h264"), static=True)
        self.streams[metric_name] = self.create_stream(fps)

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
        self.set_time(self.step)
        self.log_real_qpos(feetech_frame)
        self.step += 1

    def handle_step(self, simulation_frame: SimulationFrame):
        self.set_time(self.step)
        self.log_sim_qpos(simulation_frame)

        if simulation_frame.feetech_frame is not None:
            self.log_real_qpos(simulation_frame.feetech_frame)

        if simulation_frame.side_camera_frame is not None:
            self.log_sim_camera_frame(simulation_frame.side_camera_frame, RerunMetrics.SIM_SIDE_VIDEO_METRIC)
        if simulation_frame.link_camera_frame is not None:
            self.log_sim_camera_frame(simulation_frame.link_camera_frame, RerunMetrics.SIM_LINK_VIDEO_METRIC)

        self.step += 1

    def log_sim_qpos(self, simulation_frame: SimulationFrame):
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric("/sim/qpos", joint_name, simulation_frame.qpos[i])
            if simulation_frame.control_pos is not None:
                self.add_metric(RerunMetrics.CONTROL_POS_METRIC, joint_name, simulation_frame.control_pos[i])
            if simulation_frame.velocity is not None:
                self.add_metric("/sim/velocity", joint_name, simulation_frame.velocity[i])
            if simulation_frame.control_force is not None:
                self.add_metric("/sim/control_force", joint_name, simulation_frame.control_force[i])

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

    def log_qpos(self, step: int, worker_name: str, qpos: list[int] | list[float]):
        self.set_time(step)
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(f"/{worker_name}/qpos", joint_name, qpos[i])

    def log_control_force(self, step: int, worker_name: str, control_force: list[int] | list[float]):
        self.set_time(step)
        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            self.add_metric(f"/{worker_name}/control_force", joint_name, control_force[i])

    def log_frame(self, step: int, video_metric: str, frame: av.VideoFrame):
        # check if self.steps last element is step, only add step once
        if len(self.steps) == 0 or self.steps[-1] < step:
            self.steps.append(step)

        self.encode_frame(video_metric, frame)

    def encode_frame(self, video_metric: str, frame: av.VideoFrame):
        stream: av.VideoStream = self.streams[video_metric]
        for p in stream.encode(frame):
            packet: av.Packet = p
            self.set_time(self.steps[packet.pts]) # frames may be emitted out of order and some steps may not have a corresponding frame
            rr.log(video_metric, rr.VideoStream.from_fields(sample=bytes(packet)))

    def log_raw_frame(self, step: int, metric_name: str, frame):
        """Log a raw bitmap frame to rerun.io.

        Args:
            step: The step/frame number
            metric_name: The metric name/path to log to
            frame: Raw RGB frame as numpy array or similar (H x W x 3)
        """
        rr.log(metric_name, rr.Image(frame), static=True)

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

    def log_sim_camera_frame(self, camera_frame: CameraFrame, metric_name: str):
        frame = av.VideoFrame.from_ndarray(camera_frame.rgb, format="rgb24")
        self.log_frame(self.step, metric_name, frame)