import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator
from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase

parser = argparse.ArgumentParser(description="Run webcam capture worker 1")
parser.add_argument("--camera-id", type=int, required=True, help="Camera ID")
parser.add_argument("--width", type=int, default=640, help="Width of the webcam image")
parser.add_argument("--height", type=int, default=480, help="Height of the webcam image")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
args = parser.parse_args()

args.worker_name = WorkerBase.WORKER_WEBCAM1
args.queue_name = FifoQueue.QUEUE_WEBCAM_CAPTURE1

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_webcam_capture_worker(**vars(args))