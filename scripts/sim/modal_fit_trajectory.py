import modal
import torch

from slobot.configuration import Configuration
from slobot.lerobot.trajectory_optimizer import PytorchOptimizer

image = (
    modal.Image
    .debian_slim(
        python_version="3.12"
    ).apt_install(
        "git",
    ).uv_pip_install(
        "git+https://github.com/huggingface/lerobot.git",
        "feetech-servo-sdk",
    ).add_local_python_source(
        "slobot",
        ignore=~modal.FilePatternMatcher("**/*.py", "**/config/*")
    )
)

app = modal.App("slobot")
vol = modal.Volume.from_name("huggingface_cache", create_if_missing=True)
cache_folder = "/root/.cache/huggingface"

# modal volume put huggingface_cache ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_arm.json lerobot/calibration/robots/so100_follower/follower_arm.json

@app.function(image=image, gpu="any", timeout=3600, volumes={cache_folder: vol})
def minimize_sim_real_error(dataset_repo_id: str, episode_id: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectory_optimizer = PytorchOptimizer(
        repo_id=dataset_repo_id,
        episode_id=episode_id,
        device=device,
    )
    trajectory_optimizer.minimize_sim_real_error()
