import argparse
from slobot.teleop.asyncprocessing.workers.async_teleoperator import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Run mirror kinematics worker")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--substeps", type=int, default=40, help="Substeps")
parser.add_argument("--vis-mode", type=str, default="visual", help="Visualization mode")
parser.add_argument("--width", type=int, default=640, help="Width of the sim RGB image")
parser.add_argument("--height", type=int, default=480, help="Height of the sim RGB image")
parser.add_argument("--mjcf-path", type=str, required=True, help="Path to the MJCF file for the other robot")
parser.add_argument("--end-effector-link", type=str, required=True, help="Name of the end effector link for the other robot")
args = parser.parse_args()

async_teleoperator = AsyncTeleoperator()
async_teleoperator.spawn_mirror_kinematics_worker(**vars(args))