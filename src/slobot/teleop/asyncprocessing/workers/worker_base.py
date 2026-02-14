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
    OPERATION_MODE = OperationMode.SPAWN # use SAVE to save the worker metrics to file
    
    # Worker IDs for metrics
    WORKER_CRON = "cron"
    WORKER_LEADER = "leader"
    WORKER_FOLLOWER = "follower"
    WORKER_SIM = "sim"
    WORKER_KINEMATICS = "kinematics"
    WORKER_WEBCAM = "webcam"  # Base pattern for dynamic webcam workers (webcam1, webcam2, etc.)
    WORKER_DETECT_OBJECTS = "detect_objects"

    WORKER_NAMES = {
        WORKER_CRON: WORKER_CRON,
        WORKER_LEADER: WORKER_LEADER,
        WORKER_FOLLOWER: WORKER_FOLLOWER,
        WORKER_SIM: WORKER_SIM,
        WORKER_KINEMATICS: WORKER_KINEMATICS,
        WORKER_WEBCAM: WORKER_WEBCAM,
        WORKER_DETECT_OBJECTS: WORKER_DETECT_OBJECTS,
    }

    WORKER_INPUT_MSG_TYPE = {
        WORKER_CRON: FifoQueue.MSG_EMPTY,
        WORKER_LEADER: FifoQueue.MSG_EMPTY,
        WORKER_FOLLOWER: FifoQueue.MSG_POS,
        WORKER_SIM: FifoQueue.MSG_POS,
        WORKER_KINEMATICS: FifoQueue.MSG_POS,
        WORKER_WEBCAM: FifoQueue.MSG_EMPTY,
        WORKER_DETECT_OBJECTS: FifoQueue.MSG_OBJECT_DETECTION,
    }

    WORKER_OUTPUT_MSG_TYPE = {
        WORKER_CRON: FifoQueue.MSG_EMPTY,
        WORKER_LEADER: FifoQueue.MSG_POS,
        WORKER_FOLLOWER: FifoQueue.MSG_POS_FORCE,
        WORKER_SIM: FifoQueue.MSG_QPOS_RENDER_FORCE,
        WORKER_KINEMATICS: FifoQueue.MSG_QPOS_QPOS_RGB,
        WORKER_WEBCAM: FifoQueue.MSG_BGR,
        WORKER_DETECT_OBJECTS: FifoQueue.MSG_EMPTY,
    }

    # Worker prefixes for dynamic worker name matching
    WORKER_PREFIXES = [WORKER_WEBCAM, WORKER_DETECT_OBJECTS]

    def __init__(
        self,
        worker_name: str,
        input_queue: Optional[FifoQueue],
        output_queues: list[FifoQueue],
    ):
        """Initialize a worker.
        
        Args:
            worker_name: The worker's name
            input_queue: The queue to read input messages from (None for Cron)
            output_queues: List of queues to publish outputs to
        """
        self.worker_name = worker_name
        self.input_queue = input_queue
        self.output_queues = [queue for queue in output_queues if queue is not None]

        output_queue_names = [queue.name for queue in self.output_queues]
        self.LOGGER.info(f"Output queues for {self.worker_name}: {output_queue_names}")

        self.rerun_metrics = None

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
                match msg_type:
                    case FifoQueue.MSG_POISON_PILL:
                        self.publish_poison_pill()
                        break
                    case FifoQueue.MSG_RECORDING_ID:
                        recording_id = payload
                        self.publish_recording_id(recording_id)
                        continue

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
        self.rerun_metrics = RerunMetrics(operation_mode=WorkerBase.OPERATION_MODE, worker_name=self.worker_name)

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
        self.LOGGER.info(f"Worker {self.worker_name} received poison pill")
        for queue in self.output_queues:
            queue.send_poison_pill()

    def publish_recording_id(self, recording_id: str):
        self.rerun_metrics.init_rerun(recording_id)

        for queue in self.output_queues:
            queue.send_recording_id(recording_id)

    def publish_metrics(self, step: int, latency_ms: float):
        """Publish metrics to Rerun.io.
        
        Args:
            step: The step number
            latency_ms: Processing latency in milliseconds
        """
        self.rerun_metrics.log_latency(step, self.worker_name, latency_ms)

    def validate_input(self, msg_type: int):
        expected_msg_type = self._get_expected_input_msg_type()
        if msg_type != expected_msg_type:
            raise ValueError(f"Input type {msg_type} for worker {self.worker_name} does not match expected type {expected_msg_type}.")

    def validate_output(self, result_type: int):
        expected_msg_type = self._get_expected_output_msg_type()
        if result_type != expected_msg_type:
            raise ValueError(f"Output type {result_type} for worker {self.worker_name} does not match expected type {expected_msg_type}.")

    def _get_expected_input_msg_type(self) -> int:
        """Get expected input message type for the worker, supporting dynamic worker names."""
        # Try exact match first
        if self.worker_name in WorkerBase.WORKER_INPUT_MSG_TYPE:
            return WorkerBase.WORKER_INPUT_MSG_TYPE[self.worker_name]

        # Support dynamic workers by checking prefixes (e.g., webcam2, detect_objects4)
        for prefix in WorkerBase.WORKER_PREFIXES:
            if self.worker_name.startswith(prefix):
                return WorkerBase.WORKER_INPUT_MSG_TYPE[prefix]

        raise ValueError(f"Unknown worker name: {self.worker_name}")

    def _get_expected_output_msg_type(self) -> int:
        """Get expected output message type for the worker, supporting dynamic worker names."""
        # Try exact match first
        if self.worker_name in WorkerBase.WORKER_OUTPUT_MSG_TYPE:
            return WorkerBase.WORKER_OUTPUT_MSG_TYPE[self.worker_name]

        # Support dynamic workers by checking prefixes (e.g., webcam2, detect_objects4)
        for prefix in WorkerBase.WORKER_PREFIXES:
            if self.worker_name.startswith(prefix):
                return WorkerBase.WORKER_OUTPUT_MSG_TYPE[prefix]

        raise ValueError(f"Unknown worker name: {self.worker_name}")