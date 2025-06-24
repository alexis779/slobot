import argparse
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

parser = argparse.ArgumentParser(description="Replay a specific episode from a dataset.")
parser.add_argument("--dataset_repo_id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode_id", type=str, default=None, help="Specific episode ID or comma-separated list of IDs to replay (optional).")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG
repo_id = args.dataset_repo_id
episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path)
episode_id_str = args.episode_id
if episode_id_str is None:
    rate = episode_replayer.replay_episodes()
    print(f"Success rate: {rate:.2f}")
else:
    episode_ids = [int(eid.strip()) for eid in episode_id_str.split(',')]
    successes = []
    for episode_id in episode_ids:
        success = episode_replayer.replay_episode(episode_id)
        print(f"Episode {episode_id} success: {success}")
        successes.append(success)

    if len(successes) > 1:
        rate = sum(successes) / len(successes)
        print(f"Success rate: {rate:.2f} ({sum(successes)}/{len(successes)})")