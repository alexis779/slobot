import argparse
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

parser = argparse.ArgumentParser(description="Replay a specific episode from a dataset.")
parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode_id", type=int, required=True, help="ID of the episode to replay.")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG
repo_id = args.dataset_repo_id
episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path)
episode_replayer.replay_episode(args.episode_id)
