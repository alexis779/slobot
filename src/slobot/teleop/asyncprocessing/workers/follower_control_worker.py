"""Follower Control worker - sends control commands to the follower arm."""

from typing import Any, Optional

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.feetech import Feetech
from slobot.configuration import Configuration


class FollowerControlWorker(WorkerBase):
    """Worker that sends control commands to the follower arm.
    
    Receives qpos arrays from the leader and sends position commands to the follower.
    Publishes to three output queues:
    - sim_step_q: qpos array for simulation
    - webcam_capture_q: empty message to trigger webcam capture
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        input_queue: FifoQueue,
        webcam_capture_queues: list[FifoQueue],
        sim_step_queue: Optional[FifoQueue],
        port: str = Feetech.PORT_FOLLOWER,
    ):
        """Initialize the follower control worker.
        
        Args:
            input_queue: The queue to read qpos messages from
            webcam_capture_queues: List of queues to trigger webcam capture for multiple cameras
            sim_step_queue: Queue to send qpos for simulation
            port: Serial port for the follower arm
        """
        # Store queues for different message types
        self.webcam_capture_queues: list[FifoQueue] = webcam_capture_queues
        self.sim_step_queue: Optional[FifoQueue] = sim_step_queue

        # Combine all output queues, filtering out None values
        all_output_queues = self.webcam_capture_queues
        if sim_step_queue is not None:
            all_output_queues = all_output_queues + [sim_step_queue]

        super().__init__(
            worker_name=self.WORKER_FOLLOWER,
            input_queue=input_queue,
            output_queues=all_output_queues,
        )
        self.port = port
        self.follower: Optional[Feetech] = None

    def setup(self):
        """Initialize the follower arm connection."""
        super().setup()
        
        # Connect to follower arm with torque enabled (it's the actuator)
        self.follower = Feetech(
            port=self.port,
            robot_id=Feetech.FOLLOWER_ID,
            torque=True,
        )
        self.LOGGER.info(f"Follower arm {Feetech.FOLLOWER_ID} connected on port {self.port}")

    def teardown(self):
        """Disconnect from the follower arm."""
        self.follower.disconnect()
        
        super().teardown()

    def process(self, control_pos: list[int]) -> tuple[int, Any]:
        """Send control command to the follower arm. Then reads the motor positions.
        
        Args:
            msg_type: Should be MSG_POS
            control_pos: pos payload from leader
        
        Returns:
            Tuple of (MSG_POS_FORCE, (pos, control_force)) - the follower arm motor position
        """
        # Send control command to follower
        self.follower.control_position(control_pos)
        
        # Read follower position and convert to qpos
        follower_pos = self.follower.get_pos()

        # estimate the joint motor load by reading the control torque
        control_force = [int(x) for x in self.follower.get_dofs_control_force()]

        return FifoQueue.MSG_POS_FORCE, (follower_pos, control_force)

    def publish_data(self, step: int, result_payload: Any):
        pos, control_force = result_payload
        self.rerun_metrics.log_qpos(step, self.worker_name, pos)
        self.rerun_metrics.log_control_force(step, self.worker_name, control_force)

    def publish_outputs(self, msg_type: int, result_payload: Any, deadline: float, step: int):
        # Trigger webcam capture for all configured cameras
        for webcam_queue in self.webcam_capture_queues:
            webcam_queue.write(FifoQueue.MSG_EMPTY, None, deadline, step)

        # sends the follower pos (motor steps) as control input for the simulator
        follower_pos, _ = result_payload
        if self.sim_step_queue is not None:
            self.sim_step_queue.write(FifoQueue.MSG_POS, follower_pos, deadline, step)