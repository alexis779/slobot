import argparse

import torch

from slobot.configuration import Configuration
from slobot.lerobot.pytorch_optimizer import PytorchOptimizer

import signal
import faulthandler

# send this signal to python process to get a backtrace, for performance bottleneck analysis
faulthandler.register(signal.SIGUSR1)

parser = argparse.ArgumentParser(description="Fit the simulation trajectory to the real trajectory.")
parser.add_argument("--dataset-repo-id", type=str, required=True, help="Hugging Face Hub repository ID of the dataset.")
parser.add_argument("--episode-id", type=int, required=True, help="Episode ID to fit the trajectory to.")

args = parser.parse_args()

device = torch.device("cpu")
pytorch_optimizer = PytorchOptimizer(
    repo_id=args.dataset_repo_id,
    device=device,
)

#for episode_id in range(pytorch_optimizer.episode_loader.dataset.meta.total_episodes):
episode_id = args.episode_id
pytorch_optimizer.minimize_sim_real_error(episode_id)