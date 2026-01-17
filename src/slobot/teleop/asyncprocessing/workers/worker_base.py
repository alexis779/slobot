"""Base class for async teleoperator workers."""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics, OperationMode


class WorkerBase(ABC):
    """Base class for async teleoperator workers.
    
    Provides a message loop that polls the input queue, processes messages,
    publishes outputs, and sends metrics to the metrics queue.
    """
    
    LOGGER = Configuration.logger(__name__)

    # Operation mode for Rerun.io
    OPERATION_MODE = OperationMode.SPAWN
    
    # Worker IDs for metrics
    WORKER_CRON = "cron"
    WORKER_LEADER = "leader"
    WORKER_FOLLOWER = "follower"
    WORKER_SIM = "sim"
    WORKER_WEBCAM = "webcam"

    WORKER_NAMES = {
        WORKER_CRON: "Cron",
        WORKER_LEADER: "Leader",
        WORKER_FOLLOWER: "Follower",
        WORKER_SIM: "Sim",
        WORKER_WEBCAM: "Webcam",
    }

    WORKER_INPUT_MSG_TYPE = {
        WORKER_CRON: FifoQueue.MSG_EMPTY,
        WORKER_LEADER: FifoQueue.MSG_EMPTY,
        WORKER_FOLLOWER: FifoQueue.MSG_QPOS,
        WORKER_SIM: FifoQueue.MSG_QPOS,
        WORKER_WEBCAM: FifoQueue.MSG_EMPTY,
    }

    WORKER_OUTPUT_MSG_TYPE = {
        WORKER_CRON: FifoQueue.MSG_EMPTY,
        WORKER_LEADER: FifoQueue.MSG_QPOS,
        WORKER_FOLLOWER: FifoQueue.MSG_QPOS,
        WORKER_SIM: FifoQueue.MSG_QPOS_RGB,
        WORKER_WEBCAM: FifoQueue.MSG_BGR,
    }

    def __init__(
        self,
        worker_name: str,
        input_queue: Optional[FifoQueue],
        output_queues: list[FifoQueue],
        recording_id: str,
    ):
        """Initialize a worker.
        
        Args:
            worker_name: The worker's name
            input_queue: The queue to read input messages from (None for Cron)
            output_queues: List of queues to publish outputs to
            recording_id: The recording ID for the Rerun session
        """
        self.worker_name = worker_name
        self.input_queue = input_queue
        self.output_queues = [queue for queue in output_queues if queue is not None]

        output_queue_names = [queue.name for queue in self.output_queues]
        self.LOGGER.info(f"Output queues for {self.worker_name}: {output_queue_names}")

        self.recording_id = recording_id

        process_pid = os.getpid()
        self.LOGGER.info(f"Worker {self.worker_name} started with PID {process_pid}")

    def run(self):
        """Main worker loop. Polls input queue and processes messages."""
        self.setup()
        
        self.LOGGER.info(f"Worker {self.worker_name} started")
        
        try:
            while True:
                result = self.input_queue.poll_latest()
                
                if result is None:
                    continue
                
                msg_type, deadline, step, payload = result
                start_time = time.time()

                # Check for poison pill
                if msg_type == FifoQueue.MSG_POISON_PILL:
                    self.LOGGER.info(f"Worker {self.worker_name} received poison pill")
                    self.publish_poison_pill()
                    break

                # Validate the message
                self.validate_input(msg_type)

                # Process the message
                result_type, result_payload = self.process(payload)

                # Validate the result
                self.validate_output(result_type)

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                # Check deadline
                if end_time > deadline:
                    delay = (end_time - deadline) * 1000
                    self.LOGGER.debug(f"Worker {self.worker_name} exceeded the deadline by {delay} ms at step {step}. Latency was {latency_ms} ms.")
                
                # Publish outputs with same deadline (time remaining decreases as we progress) and step
                self.publish_outputs(result_type, result_payload, deadline, step)

                # Publish data
                self.publish_data(step, result_payload)

                # Publish metrics
                self.publish_metrics(step, latency_ms)
                
        except Exception as e:
            self.LOGGER.error(f"Worker {self.worker_name} error: {e}")
            raise
        finally:
            self.teardown()
            self.LOGGER.info(f"Worker {self.worker_name} stopped")

    def setup(self):
        """Called once before the main loop. Override to initialize resources."""
        self.setup_input()

        self.setup_output()

        self.setup_metrics()

    def setup_input(self):
        self.input_queue.open_read()

    def setup_output(self):
        """Open output queues for writing."""
        for queue in self.output_queues:
            queue.open_write()

    def setup_metrics(self):
        self.rerun_metrics = RerunMetrics(recording_id=self.recording_id, operation_mode=WorkerBase.OPERATION_MODE)
        self.add_worker_metric_labels()

    def add_worker_metric_labels(self):
        for worker_name in WorkerBase.WORKER_NAMES.values():
            self.rerun_metrics.add_child_metric_label(f"/latency", worker_name, f"{worker_name} latency (ms)")

    def teardown(self):
        """Called once after the main loop. Override to cleanup resources."""
        self.input_queue.close()
        
        for queue in self.output_queues:
            queue.close()

    @abstractmethod
    def process(self, payload: Any) -> tuple[int, Any]:
        """Process an input message and return the output.
        
        Args:
            payload: The input payload
        
        Returns:
            Tuple of (output_msg_type, output_payload)
        """
        raise NotImplementedError

    @abstractmethod
    def publish_data(self, step: int, result_payload: Any):
        """Publish data to Rerun.io."""
        raise NotImplementedError

    def publish_outputs(self, msg_type: int, result_payload: Any, deadline: float, step: int):
        """Publish outputs to all output queues.
        
        Override this method if different queues need different message types.
        The deadline is propagated unchanged - downstream workers have less time remaining.
        
        Args:
            msg_type: The output message type
            result_payload: The output payload
            deadline: The deadline for downstream processing (propagated from input)
        """
        for queue in self.output_queues:
            queue.write(msg_type, result_payload, deadline, step)

    def publish_poison_pill(self):
        """Publish a poison pill message to signal graceful shutdown to downstream workers."""
        for queue in self.output_queues:
            queue.send_poison_pill()

    def publish_metrics(self, step: int, latency_ms: float):
        """Publish metrics to Rerun.io.
        
        Args:
            step: The step number
            latency_ms: Processing latency in milliseconds
        """
        self.rerun_metrics.log_latency(step, self.worker_name, latency_ms)

    def validate_input(self, msg_type: int):
        expected_msg_type = WorkerBase.WORKER_INPUT_MSG_TYPE[self.worker_name]
        if msg_type != expected_msg_type:
            raise ValueError(f"Input type {msg_type} for worker {self.worker_name} does not match expected type {expected_msg_type}.")

    def validate_output(self, result_type: int):
        expected_msg_type = WorkerBase.WORKER_OUTPUT_MSG_TYPE[self.worker_name]
        if result_type != expected_msg_type:
            raise ValueError(f"Output type {result_type} for worker {self.worker_name} does not match expected type {expected_msg_type}.")