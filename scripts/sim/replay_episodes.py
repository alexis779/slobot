import argparse
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

parser = argparse.ArgumentParser(description="Replay a specific episode from a dataset.")
parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode_id", type=str, default=None, help="Specific episode ID or comma-separated list of IDs to replay (optional).")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG
repo_id = args.dataset_repo_id
episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path, show_viewer=True)


episode_ids = None

episode_id_str = args.episode_id
if episode_id_str is not None:
    episode_ids = [ int(episode_id)
        for episode_id in episode_id_str.split(',')
    ]

rate = episode_replayer.replay_episodes(episode_ids)
print(f"Success rate: {rate:.2f}")