import argparse

import torch

from slobot.configuration import Configuration
from slobot.lerobot.pytorch_optimizer import PytorchOptimizer

parser = argparse.ArgumentParser(description="Fit the simulation trajectory to the real trajectory.")
parser.add_argument("--dataset-repo-id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode-id", type=int, required=True, help="Specific episode ID to fit the trajectory.")

args = parser.parse_args()

mjcf_path = Configuration.MJCF_CONFIG

device = torch.device("cpu")
optimizer = PytorchOptimizer(
    repo_id=args.dataset_repo_id,
    mjcf_path=mjcf_path,
    episode_id=args.episode_id,
    device=device,
)
optimizer.minimize_sim_real_error()