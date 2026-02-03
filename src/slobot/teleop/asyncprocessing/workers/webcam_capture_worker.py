"""Webcam Capture worker - captures frames from the webcam."""

from typing import Any, Optional
import enum

import cv2
import av
import rerun as rr
import numpy as np

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.configuration import Configuration
from slobot.teleop.asyncprocessing.shared_memory_block import SharedMemoryBlock


class DetectionTask(enum.Enum):
    DETECT = "detect", "yolo26n.pt"
    POSE = "pose", "yolo26n-pose.pt"

    def __init__(self, value, model_name):
        self._value_ = value
        self.model_name = model_name


class WebcamCaptureWorker(WorkerBase):
    """Worker that captures frames from the webcam.
    
    Receives empty tick messages and captures a frame from the webcam.
    Publishes the RGB image to metrics.
    Writes frame to Shared Memory for decoupled detection.
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
        detect_objects_queue: Optional[FifoQueue] = None,
    ):
        """Initialize the webcam capture worker.
        
        Args:
            worker_name: The name of the worker (e.g., "webcam2", "webcam4")
            input_queue: The queue to read tick messages from
            camera_id: The camera device ID (0 for default webcam)
            width: Width of the webcam image
            height: Height of the webcam image
            fps: Height of the webcam image
            detect_objects_queue: Optional queue to signal detection worker
        """
        super().__init__(
            worker_name=worker_name,
            input_queue=input_queue,
            output_queues=[detect_objects_queue],  # No downstream workers
        )
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.detect_objects_queue = detect_objects_queue
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[YOLO] = None
        self.shm_block: Optional[SharedMemoryBlock] = None

    def setup(self):
        """Initialize the webcam capture."""
        super().setup()

        # Shared memory will be initialized in setup if needed


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

        if self.detect_objects_queue:
             # Calculate size: W x H x C (BGR) + header
             size = SharedMemoryBlock.HEADER_SIZE + self.width * self.height * SharedMemoryBlock.CHANNELS

             # Use centralized naming convention
             shm_name = SharedMemoryBlock.get_name_from_camera_id(self.camera_id)
             self.shm_block = SharedMemoryBlock.create(shm_name, size)

    def teardown(self):
        """Release the webcam."""
        if self.cap:
            self.cap.release()

        if self.shm_block:
            self.shm_block.close()
            self.shm_block.unlink()

        super().teardown()

    def _write_frame_to_shm(self, frame: np.ndarray):
        """Write frame to shared memory and signal detection worker."""
        if not self.shm_block.write_frame(frame):
            return

        # Signal detection worker
        self.detect_objects_queue.write(FifoQueue.MSG_OBJECT_DETECTION, b'', 0.0, 0)

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
            raise RuntimeError("Failed to capture frame from webcam")

        # Write to shared memory
        if self.detect_objects_queue:
            self._write_frame_to_shm(frame)

        return FifoQueue.MSG_BGR, frame

    def publish_outputs(self, msg_type: int, result_payload: Any, deadline: float, step: int):
        # Trigger detect objects
        if self.detect_objects_queue:
            self.detect_objects_queue.write(FifoQueue.MSG_EMPTY, None, deadline, step)

    def publish_recording_id(self, recording_id: str):
        super().publish_recording_id(recording_id)
        self.rerun_metrics.add_video_stream(f"/{self.worker_name}/video")

    def publish_data(self, step: int, bgr: Any):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        self.rerun_metrics.log_frame(step, f"/{self.worker_name}/video", frame, self.stream)