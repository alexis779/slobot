import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run cron worker")
parser.add_argument("--recording-id", type=str, required=True, help="The rerun recording id")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_cron_worker(**vars(args))