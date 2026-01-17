import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run sim step worker")
parser.add_argument("--recording-id", type=str, required=True, help="The rerun recording id")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--substeps", type=int, default=40, help="Substeps")
parser.add_argument("--vis-mode", type=str, default="visual", help="Visualization mode")
parser.add_argument("--width", type=int, default=640, help="Width of the sim RGB image")
parser.add_argument("--height", type=int, default=480, help="Height of the sim RGB image")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_sim_step_worker(**vars(args))