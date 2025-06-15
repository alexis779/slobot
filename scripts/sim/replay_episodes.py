import argparse
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

parser = argparse.ArgumentParser(description="Replay a specific episode from a dataset.")
parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode_id", type=int, default=None, help="Specific episode ID to replay (optional).")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG
repo_id = args.dataset_repo_id
episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path)
episode_id = args.episode_id
if episode_id is None:
    score = episode_replayer.replay_episodes()
    print("score=", score)
else:
    success = episode_replayer.replay_episode(episode_id)
    print("success=", success)