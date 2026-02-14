"""Sim Step worker - runs the Genesis simulation step."""

from typing import Any

import torch
import av

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100
from slobot.video_streams import VideoStreams

from enum import Enum

class RenderMode(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"
    NORMAL = "normal"


class SimStepWorker(WorkerBase):
    """Worker that runs the Genesis simulation step.
    
    Receives qpos arrays and runs a simulation step with that control input.
    Publishes the resulting qpos and RGB render to metrics.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        input_queue: FifoQueue,
        fps: int,
        substeps: int,
        vis_mode: str,
        width: int,
        height: int,
    ):
        """Initialize the sim step worker.
        
        Args:
            input_queue: The queue to read qpos messages from
            fps: Expected frames per second
            substeps: Number of substeps
            vis_mode: Visualization mode
            width: Width of the sim RGB image
            height: Height of the sim RGB image
        """
        super().__init__(
            worker_name=WorkerBase.WORKER_SIM,
            input_queue=input_queue,
            output_queues=[],  # No downstream workers
        )
        self.fps = fps
        self.substeps = substeps
        self.vis_mode = vis_mode
        self.width = width
        self.height = height

    def setup(self):
        """Initialize the Genesis simulation."""
        super().setup()

        # Feetech instance for pos (motor steps) to qpos (radians) conversion
        self.feetech = Feetech(connect=False)

        for render_mode in RenderMode:
            self.rerun_metrics.add_video_stream(self.metric_name(render_mode))

        # initialize the video streams
        self.container = av.open("/dev/null", "w", format="h264")
        self.streams = {
            render_mode: self.create_stream() for render_mode in RenderMode
        }

        res = (self.width, self.height)
        self.arm = SoArm100(show_viewer=False, fps=self.fps, substeps=self.substeps, rgb=True, depth=True, segmentation=True, normal=True, res=res, vis_mode=self.vis_mode)
        
        self.LOGGER.info(f"Genesis simulation started with {self.fps} FPS, {self.substeps} substeps, {self.width}x{self.height} resolution, and {self.vis_mode} visualization mode")

    def teardown(self):
        # flush video streams
        for render_mode in RenderMode:
           self.flush_stream(render_mode)

        # Close the container
        self.container.close()

        # Stop the Genesis simulation
        self.arm.genesis.stop()
        
        super().teardown()

    def process(self, control_pos: list[int]) -> tuple[int, Any]:
        """Run a simulation step with the given control input.
        
        Args:
            control_pos: Motor positions in steps (from MSG_POS)
        
        Returns:
            Tuple of (MSG_QPOS_RENDER_FORCE, (qpos, rgb, ...)) - the simulated qpos and RGB image
        """
        # Convert motor steps to qpos (radians) for Genesis
        control_qpos = self.feetech.pos_to_qpos(control_pos)
        control_qpos = torch.tensor([control_qpos], dtype=torch.float32)
        self.arm.genesis.entity.control_dofs_position(control_qpos)
        
        # Step the simulation
        self.arm.genesis.step()
        
        # Get the resulting qpos
        qpos = self.arm.genesis.entity.get_qpos()
        qpos = qpos[0].tolist()

        # estimate the joint motor load by reading the control torque
        control_force = self.arm.genesis.entity.get_dofs_control_force()
        control_force = control_force[0].tolist()
        
        # Render the camera
        rgb, depth, segmentation, normal = self.arm.genesis.side_camera.render(rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True)
        
        return FifoQueue.MSG_QPOS_RENDER_FORCE, (qpos, rgb, depth, segmentation, normal, control_force)

    def publish_data(self, step: int, result_payload: Any):
        qpos, rgb, depth, segmentation, normal, control_force = result_payload

        self.rerun_metrics.log_qpos(step, self.worker_name, qpos)
        self.rerun_metrics.log_control_force(step, self.worker_name, control_force)

        self.log_rgb(step, rgb, RenderMode.RGB)

        depth = VideoStreams.logarithmic_depth_to_rgb(depth)
        self.log_rgb(step, depth, RenderMode.DEPTH)

        self.log_rgb(step, segmentation, RenderMode.SEGMENTATION)

        self.log_rgb(step, normal, RenderMode.NORMAL)

    def publish_recording_id(self, recording_id: str):
        super().publish_recording_id(recording_id)
        for render_mode in RenderMode:
            self.rerun_metrics.add_video_stream(f"/{self.worker_name}/{render_mode.value}/video")

    def log_rgb(self, step: int, rgb: Any, render_mode: RenderMode):
        # transcode image into a video stream to reduce disk space
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        self.rerun_metrics.log_frame(step, self.render_mode_metric_name(render_mode), frame, self.streams[render_mode])

    def flush_stream(self, render_mode: RenderMode):
        self.rerun_metrics.encode_frame(self.render_mode_metric_name(render_mode), None, self.streams[render_mode])

    def render_mode_metric_name(self, render_mode: RenderMode):
        return f"/{self.worker_name}/{render_mode.value}/video"

    def create_stream(self):
        stream = self.container.add_stream("libx264", rate=self.fps)
        stream.max_b_frames = 0
        return stream