import argparse

from slobot.teleop.recording_replayer import RecordingReplayer

parser = argparse.ArgumentParser(description="Replay a teleoperation recording")
parser.add_argument("--rrd-file", type=str, required=True, help="Path to the .rrd recording file")
parser.add_argument("--recording-id", type=str, required=True, help="The id of the recording to generate")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for replay")
parser.add_argument("--substeps", type=int, default=1, help="Substeps per step for replay")
parser.add_argument("--vis-mode", type=str, default='visual', help="Visualization mode for replay")
parser.add_argument("--golf-ball-pos", type=str, default=None, help="Initial position of the golf ball")
parser.add_argument("--diff-threshold", type=int, default=200, help="Threshold for the difference between the leader and follower gripper motor positions")
args = parser.parse_args()

recording_replayer = RecordingReplayer(**vars(args))
recording_replayer.replay()