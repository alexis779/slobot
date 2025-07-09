from slobot.lerobot.sim_dataset_generator import SimDatasetGenerator
from slobot.configuration import Configuration
import argparse

parser = argparse.ArgumentParser(description="Generate a sim dataset from a dataset.")
parser.add_argument("--dataset-repo-id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--sim-dataset-repo-id", type=str, required=True, help="Hugging Face Hub repository ID of the sim dataset.")
parser.add_argument("--episode-ids", type=str, required=False, help="Comma-separated list of episode IDs to process (e.g., 1,2,3)")

args = parser.parse_args()

episode_ids = None
if args.episode_ids:
    episode_ids = [int(episode_id_str) for episode_id_str in args.episode_ids.split(",")]

mjcf_path = Configuration.MJCF_CONFIG

sim_dataset_generator = SimDatasetGenerator(repo_id=args.dataset_repo_id, sim_repo_id=args.sim_dataset_repo_id, mjcf_path=mjcf_path)
sim_dataset_generator.generate_dataset(episode_ids=episode_ids)