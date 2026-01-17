"""Leader Read worker - reads the leader arm position."""

from types import NoneType
from typing import Any, Optional

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.feetech import Feetech
from slobot.configuration import Configuration


class LeaderReadWorker(WorkerBase):
    """Worker that reads the leader arm position.
    
    Receives empty tick messages and publishes the leader arm's joint positions
    as qpos arrays.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        input_queue: FifoQueue,
        follower_control_queue: FifoQueue,
        recording_id: str,
        port: str = Feetech.PORT1,
    ):
        """Initialize the leader read worker.
        
        Args:
            input_queue: The queue to read tick messages from
            follower_control_queue: The queue to publish qpos to (typically follower_control_q)
            recording_id: The rerun recording id
            port: Serial port for the leader arm
        """
        super().__init__(
            worker_name=self.WORKER_LEADER,
            input_queue=input_queue,
            output_queues=[follower_control_queue],
            recording_id=recording_id,
        )
        self.port = port
        self.leader: Optional[Feetech] = None

    def setup(self):
        """Initialize the leader arm connection."""
        super().setup()
        
        # Connect to leader arm with torque disabled (it's the input device)
        self.leader = Feetech(
            port=self.port,
            robot_id=Feetech.LEADER_ID,
            torque=False,
        )
        self.LOGGER.info(f"Leader arm {Feetech.LEADER_ID} connected on port {self.port}")

    def teardown(self):
        """Disconnect from the leader arm."""
        self.leader.disconnect()
        
        super().teardown()

    def process(self, payload: NoneType) -> tuple[int, list[float]]:
        """Read the leader arm position.
        
        Args:
            msg_type: Should be MSG_EMPTY (tick)
            payload: None
        
        Returns:
            Tuple of (MSG_QPOS, qpos_payload)
        """
        # Read leader position and convert to qpos
        leader_pos = self.leader.get_pos()
        leader_qpos = self.leader.pos_to_qpos(leader_pos)
        
        return FifoQueue.MSG_QPOS, leader_qpos

    def publish_data(self, step: int, leader_qpos: list[float]):
        self.rerun_metrics.log_qpos(step, self.worker_name, leader_qpos)
