import argparse

from slobot.teleop.recording_replayer import RecordingReplayer

parser = argparse.ArgumentParser(description="Replay a teleoperation recording")
parser.add_argument("--rrd_file", type=str, required=True, help="Path to the .rrd recording file")
parser.add_argument("--fps", type=int, default=10, help="Frames per second for replay")
args = parser.parse_args()

recording_replayer = RecordingReplayer(**vars(args))
recording_replayer.replay()