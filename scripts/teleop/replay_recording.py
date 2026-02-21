import argparse

from slobot.teleop.recording_replayer import RecordingReplayer
from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.configuration import Configuration

parser = argparse.ArgumentParser(description="Replay a teleoperation recording")
parser.add_argument("--rrd-file", type=str, required=True, help="Path to the .rrd recording file")
parser.add_argument("--recording-id", type=str, required=True, help="The id of the recording to generate")
parser.add_argument("--pick-frame-id", type=int, help="The frame id to pick the ball")
args = parser.parse_args()

golf_ball_env = GolfBallEnv()
recording_replayer = RecordingReplayer(golf_ball_env=golf_ball_env, diff_threshold=Configuration.DIFF_THRESHOLD)
recording_replayer.replay(args.rrd_file, args.pick_frame_id)