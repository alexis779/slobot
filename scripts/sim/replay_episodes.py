import argparse
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer, GridSearchInput, MotorRangeInput

parser = argparse.ArgumentParser(description="Replay a specific episode from a dataset.")
parser.add_argument("--dataset-repo-id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode-ids", type=str, default=None, help="Specific episode ID or comma-separated list of IDs to replay (optional).")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG
repo_id = args.dataset_repo_id

episode_count = None
episode_ids = None
episode_id_str = args.episode_ids
if episode_id_str is not None:
    episode_ids = [ int(episode_id)
        for episode_id in episode_id_str.split(',')
    ]
    episode_count = len(episode_ids)

episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path, show_viewer=False, n_envs=episode_count)
episode_replayer.load_episodes(episode_ids=episode_ids)
rate = episode_replayer.replay_episodes()
print(f"Success rate: {rate:.2f} for episode_ids = {episode_ids}")