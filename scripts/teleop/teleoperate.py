import argparse
from slobot.teleop.teleoperator import Teleoperator

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--recording_id', type=str)
args = parser.parse_args()

teleoperator = Teleoperator(**vars(args))
teleoperator.teleoperate()