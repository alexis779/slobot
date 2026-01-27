import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run follower control worker")
parser.add_argument("--port", type=str, required=True, help="Follower port")
parser.add_argument("--webcam1", action="store_true", default=False, help="Enable webcam1 capture")
parser.add_argument("--webcam2", action="store_true", default=False, help="Enable webcam2 capture")
parser.add_argument("--sim", action="store_true", default=False, help="Enable simulation")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_follower_control_worker(**vars(args))