import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run follower control worker")
parser.add_argument("--port", type=str, required=True, help="Follower port")
parser.add_argument("--camera-id", type=int, action="append", dest="camera_ids", help="Camera ID to enable (can be specified multiple times)")
parser.add_argument("--sim", action="store_true", default=False, help="Enable simulation")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_follower_control_worker(**vars(args))