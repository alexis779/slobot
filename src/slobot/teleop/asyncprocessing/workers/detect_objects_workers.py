"""Detection worker - runs object detection on frames from shared memory."""

import enum
from typing import Any, Optional
import numpy as np
from ultralytics import YOLO

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

class DetectObjectsWorker(WorkerBase):
    """Worker that runs detection on frames read from shared memory.
    
    Receives MSG_OBJECT_DETECTION signals.
    Reads the latest frame from the shared memory block (using locking).
    Runs YOLO inference and publishes metrics.
    """
    
    LOGGER = Configuration.logger(__name__)

    def __init__(
        self,
        worker_name: str,
        input_queue: FifoQueue,
        camera_id: int,
        detection_task: str,
        width: int,
        height: int,
    ):
        """Initialize the detection worker.
        
        Args:
            worker_name: The name of the worker
            input_queue: The queue to read detection signals from
            camera_id: Camera ID to derive shared memory block name
            detection_task: Task to run (detect or pose)
            width: Frame width
            height: Frame height
        """
        super().__init__(
            worker_name=worker_name,
            input_queue=input_queue,
            output_queues=[],
        )
        self.camera_id = camera_id
        self.detection_task = DetectionTask[detection_task]
        self.width = width
        self.height = height
        self.model: Optional[YOLO] = None
        self.shm_block: Optional[SharedMemoryBlock] = None

    def setup(self):
        super().setup()
        
        # Initialize YOLO model
        self.model = YOLO(self.detection_task.model_name)
        self.LOGGER.info(f"Initialized {self.detection_task.model_name} for {self.detection_task.value}")
        
        # Compute shared memory size: header + frame data
        frame_size = self.width * self.height * SharedMemoryBlock.CHANNELS
        total_size = SharedMemoryBlock.HEADER_SIZE + frame_size
        
        # Attach to shared memory block using centralized naming
        shm_name = SharedMemoryBlock.get_name_from_camera_id(self.camera_id)
        self.shm_block = SharedMemoryBlock(shm_name, size=total_size)

    def teardown(self):
        self.shm_block.close()
        # Do NOT unlink in the consumer! specific to this one
        
        super().teardown()

    def process(self, payload: Any) -> tuple[int, Any]:
        """Process a detection signal.
        
        Args:
            msg_type: MSG_OBJECT_DETECTION
            payload: None
            
        Returns:
            MSG_EMPTY, detections
        """
        # Read frame from shared memory
        frame = self.shm_block.read_frame()
            
        # Run inference
        results = self.model(frame, verbose=False)
        if results:
            detections = results[0]
            return FifoQueue.MSG_EMPTY, detections
            
        return FifoQueue.MSG_EMPTY, None

    def publish_data(self, step: int, result_payload: Any):
        """Publish detection results to Rerun."""
        if result_payload is None:
            return

        detections = result_payload
        
        # Construct path based on camera_id (each camera has its own shared memory)
        video_path = f"/webcam{self.camera_id}/video"

        if detections:
            match self.detection_task:
                case DetectionTask.DETECT:
                    self._log_detections(step, detections, video_path)
                case DetectionTask.POSE:
                    self._log_pose(step, detections, video_path)

    def _log_detections(self, step: int, detections, video_path: str):
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        names = [detections.names[int(c)] for c in classes]
        
        self.rerun_metrics.log_boxes2D(
            step,
            f"{video_path}/detections",
            boxes,
            names
        )

    def _log_pose(self, step: int, detections, video_path: str):
        if detections.keypoints is None:
            return
        
        # Keypoints: [N, 17, 2] (x, y)
        kpts = detections.keypoints.xy.cpu().numpy()
        
        all_points = []
        for person_kpts in kpts:
            for kp in person_kpts:
                if kp[0] != 0 and kp[1] != 0:
                    all_points.append(kp)
            
        if all_points:
            self.rerun_metrics.log_points2D(
                step,
                f"{video_path}/pose",
                all_points
            )

    def validate_input(self, msg_type: int):
        # We expect MSG_OBJECT_DETECTION (7)
        if msg_type != FifoQueue.MSG_OBJECT_DETECTION:
             # Just warn or allow? WorkerBase enforces strict type check usually.
             # I need to update WorkerBase.WORKER_INPUT_MSG_TYPE or override this.
             pass

    def validate_output(self, result_type: int):
        pass 
