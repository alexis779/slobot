"""Webcam Capture worker - captures frames from the webcam."""

from typing import Any, Optional

import cv2
import av

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
        worker_name: str,
        input_queue: FifoQueue,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
    ):
        """Initialize the webcam capture worker.
        
        Args:
            worker_name: The name of the worker
                (either WORKER_WEBCAM1 or WORKER_WEBCAM2)
            input_queue: The queue to read tick messages from
            camera_id: The camera device ID (0 for default webcam)
            width: Width of the webcam image
            height: Height of the webcam image
        """
        super().__init__(
            worker_name=worker_name,
            input_queue=input_queue,
            output_queues=[],  # No downstream workers
        )
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def setup(self):
        """Initialize the webcam capture."""
        super().setup()

        # initialize the video stream
        container = av.open("/dev/null", "w", format="h264")
        self.stream = container.add_stream("libx264", rate=self.fps)

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

    def publish_recording_id(self, recording_id: str):
        super().publish_recording_id(recording_id)
        self.rerun_metrics.add_video_stream(f"/{self.worker_name}/video")

    def publish_data(self, step: int, bgr: Any):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        self.rerun_metrics.log_frame(step, f"/{self.worker_name}/video", frame, self.stream)