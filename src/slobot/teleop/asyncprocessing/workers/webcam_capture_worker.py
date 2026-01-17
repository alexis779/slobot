"""Webcam Capture worker - captures frames from the webcam."""

from typing import Any, Optional

import cv2
import rerun as rr

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.configuration import Configuration


class WebcamCaptureWorker(WorkerBase):
    """Worker that captures frames from the webcam.
    
    Receives empty tick messages and captures a frame from the webcam.
    Publishes the RGB image to metrics.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        input_queue: FifoQueue,
        recording_id: str,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
    ):
        """Initialize the webcam capture worker.
        
        Args:
            input_queue: The queue to read tick messages from
            recording_id: The recording ID for the Rerun session
            camera_id: The camera device ID (0 for default webcam)
            width: Width of the webcam image
            height: Height of the webcam image
        """
        super().__init__(
            worker_name=self.WORKER_WEBCAM,
            input_queue=input_queue,
            output_queues=[],  # No downstream workers
            recording_id=recording_id,
        )
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def setup(self):
        """Initialize the webcam capture."""
        super().setup()
        
        # Open the webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Set format to MJPG
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.LOGGER.info(f"Webcam {self.camera_id} opened with resolution {actual_width}x{actual_height} @ {actual_fps} FPS")

    def teardown(self):
        """Release the webcam."""
        self.cap.release()
        
        super().teardown()

    def process(self, payload: Any) -> tuple[int, Any]:
        """Capture a frame from the webcam.
        
        Args:
            msg_type: Should be MSG_EMPTY (tick)
            payload: Empty payload
        
        Returns:
            Tuple of (MSG_RGB, rgb_payload)
        """
        # Capture frame
        ret, frame = self.cap.read()
        
        if not ret:
            self.LOGGER.warning("Failed to capture frame from webcam")
            return FifoQueue.MSG_EMPTY, b''

        return FifoQueue.MSG_BGR, frame

    def publish_data(self, step: int, frame: Any):
        self.rerun_metrics.log_bgr(step, self.worker_name, frame)