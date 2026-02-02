import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator
from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue

parser = argparse.ArgumentParser(description="Run object detection worker")
parser.add_argument("--camera-id", type=int, required=True, help="Camera ID")
parser.add_argument("--detection-task", type=str, required=True, help="Detection task (detect or pose)")
parser.add_argument("--width", type=int, required=True, help="Frame width")
parser.add_argument("--height", type=int, required=True, help="Frame height")
args = parser.parse_args()

# Create dynamic worker and queue names based on camera ID
args.worker_name = f"detect_objects{args.camera_id}"
args.queue_name = FifoQueue.get_queue_name(FifoQueue.QUEUE_OBJECT_DETECTION, args.camera_id)

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_detect_objects_worker(**vars(args))

