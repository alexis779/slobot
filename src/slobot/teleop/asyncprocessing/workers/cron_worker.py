"""Cron worker - publishes ticks at a fixed rate."""

import time
from types import NoneType
from typing import Any
import signal
from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.configuration import Configuration


class CronWorker(WorkerBase):
    """Cron-like worker that publishes ticks at a fixed rate (e.g., 30 Hz).
    
    This is the main scheduler that initiates each cycle of the control loop.
    Unlike other workers, it doesn't have an input queue - it generates ticks
    based on a timer.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        leader_read_queue: FifoQueue,
        recording_id: str,
        fps: int,
    ):
        """Initialize the cron worker.
        
        Args:
            leader_read_queue: The queue to publish ticks to (typically leader_read_q)
            fps: Target frequency in Hz
        """
        super().__init__(
            worker_name=self.WORKER_CRON,
            input_queue=None,
            output_queues=[leader_read_queue],
            recording_id=recording_id,
        )

        self.leader_read_queue = leader_read_queue
        self.period = 1.0 / fps

    def setup(self):
        self.setup_output()
        self.setup_metrics()
        self.add_signal_handlers()
        self.running = True # start the workflow at startup
        self.shutdown = False # shutdown flag

    def run(self):
        """Main cron loop. Publishes ticks at the configured rate."""
        self.setup()

        self.LOGGER.info(f"Cron worker started with period {self.period} seconds")

        try:
            step = 0
            while True:
                if self.shutdown:
                    break

                start_time = time.time()
                
                # Set deadline = start_time + period
                # All downstream workers must complete before this deadline
                deadline = start_time + self.period
    
                if self.running:
                    # Publish tick to all output queues with the deadline
                    self.leader_read_queue.write_empty(deadline, step)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                if self.running:
                    # Publish metrics
                    self.publish_metrics(step, latency_ms)
                
                # Sleep for remaining time in the period
                elapsed = end_time - start_time
                sleep_time = self.period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1
                    
        except Exception as e:
            self.LOGGER.error(f"Cron worker error: {e}")
            raise
        finally:
            self.teardown()
            self.LOGGER.info("Cron worker stopped")


    def add_signal_handlers(self):
        signal.signal(signal.SIGINT, self.stop_workers)
        signal.signal(signal.SIGTERM, self.stop_workers)
        signal.signal(signal.SIGUSR1, self.start_workers)
        signal.signal(signal.SIGUSR2, self.pause_workers)

    def stop_workers(self, signum, frame):
        self.LOGGER.info(f"Stopping workers due to signal {signum}.")
        self.publish_poison_pill()
        self.shutdown = True

    def start_workers(self, signum, frame):
        self.LOGGER.info(f"Starting workers due to signal {signum}.")
        self.running = True

    def pause_workers(self, signum, frame):
        self.LOGGER.info(f"Pausing workers due to signal {signum}.")
        self.running = False

    def process(self, payload: Any) -> tuple[int, Any]:
        return FifoQueue.MSG_EMPTY, None

    def publish_data(self, step: int, result_payload: NoneType):
        pass

    def teardown(self):
        """Close output queues."""
        self.shutdown = True
        for queue in self.output_queues:
            queue.close()