import argparse

from slobot.teleop.recording_replayer import RecordingReplayer

parser = argparse.ArgumentParser(description="Replay a teleoperation recording")
parser.add_argument("--rrd_file", type=str, required=True, help="Path to the .rrd recording file")
parser.add_argument("--fps", type=int, default=24, help="Frames per second for replay")
parser.add_argument("--substeps", type=int, default=1, help="Substeps per step for replay")
parser.add_argument("--vis_mode", type=str, default='visual', help="Visualization mode for replay")
parser.add_argument("--golf_ball_pos", type=str, default=None, help="Initial position of the golf ball")
args = parser.parse_args()

recording_replayer = RecordingReplayer(**vars(args))
recording_replayer.replay()