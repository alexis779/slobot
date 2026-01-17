import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run webcam capture worker")
parser.add_argument("--recording-id", type=str, required=True, help="The rerun recording id")
parser.add_argument("--camera-id", type=int, required=True, help="Camera ID")
parser.add_argument("--width", type=int, default=640, help="Width of the webcam image")
parser.add_argument("--height", type=int, default=480, help="Height of the webcam image")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_webcam_capture_worker(**vars(args))