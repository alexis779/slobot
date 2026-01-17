"""Sim Step worker - runs the Genesis simulation step."""

import struct
from typing import Any, Optional

import numpy as np
import torch

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100


class SimStepWorker(WorkerBase):
    """Worker that runs the Genesis simulation step.
    
    Receives qpos arrays and runs a simulation step with that control input.
    Publishes the resulting qpos and RGB render to metrics.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        input_queue: FifoQueue,
        recording_id: str,
        fps: int,
        substeps: int,
        vis_mode: str,
        width: int,
        height: int,
    ):
        """Initialize the sim step worker.
        
        Args:
            input_queue: The queue to read qpos messages from
            recording_id: The recording ID for the Rerun session
            fps: Expected frames per second
            substeps: Number of substeps
            vis_mode: Visualization mode
            width: Width of the sim RGB image
            height: Height of the sim RGB image
        """
        super().__init__(
            worker_name=self.WORKER_SIM,
            input_queue=input_queue,
            output_queues=[],  # No downstream workers
            recording_id=recording_id,
        )
        self.fps = fps
        self.substeps = substeps
        self.vis_mode = vis_mode
        self.width = width
        self.height = height

    def setup(self):
        """Initialize the Genesis simulation."""
        super().setup()

        res = (self.width, self.height)
        self.arm = SoArm100(show_viewer=False, fps=self.fps, substeps=self.substeps, rgb=True, res=res, vis_mode=self.vis_mode)
        
        self.LOGGER.info(f"Genesis simulation started with {self.fps} FPS, {self.substeps} substeps, {self.width}x{self.height} resolution, and {self.vis_mode} visualization mode")

    def teardown(self):
        """Stop the Genesis simulation."""
        self.arm.genesis.stop()
        
        super().teardown()

    def process(self, control_qpos: list[float]) -> tuple[int, Any]:
        """Run a simulation step with the given control input.
        
        Args:
            msg_type: Should be MSG_QPOS
            payload: qpos payload with target joint positions
        
        Returns:
            Tuple of (MSG_QPOS_RGB, (qpos_payload, rgb_payload)) - the simulated qpos and RGB image
        """
        # Convert to tensor and apply control
        control_qpos = torch.tensor([control_qpos], dtype=torch.float32)
        self.arm.genesis.entity.control_dofs_position(control_qpos)
        
        # Step the simulation
        self.arm.genesis.step()
        
        # Get the resulting qpos
        qpos = self.arm.genesis.entity.get_qpos()
        qpos = qpos[0].tolist()
        
        # Render the camera
        rgb, _, _, _ = self.arm.genesis.camera.render()
        
        return FifoQueue.MSG_QPOS_RGB, (qpos, rgb)

    def publish_data(self, step: int, result_payload: Any):
        qpos, frame = result_payload

        self.rerun_metrics.log_qpos(step, self.worker_name, qpos)
        self.rerun_metrics.log_rgb(step, self.worker_name, frame)