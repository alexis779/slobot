import argparse
from slobot.teleop.teleoperator import Teleoperator

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, required=True)
args = parser.parse_args()

teleoperator = Teleoperator(fps=args.fps)
teleoperator.teleoperate()