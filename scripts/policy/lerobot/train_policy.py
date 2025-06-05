from pathlib import Path
import modal
from huggingface_hub import upload_folder

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.scripts.train import init_logging, train, TrainPipelineConfig
from lerobot.common.policies.factory import make_policy_config

image = (
    modal.Image
    .debian_slim()
    .apt_install(
        "git",
        "ffmpeg",
    )
    .pip_install(
        "git+https://github.com/huggingface/lerobot.git",
        "transformers",
        "num2words"
    )
)

app = modal.App("lerobot")
vol = modal.Volume.from_name("lerobot", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")
hf_secret = modal.Secret.from_name("hf-secret")

partition = "/data"

def get_output_dir(dataset_repo_id: str, policy_type: str):
    return Path(partition, "outputs", "train", dataset_repo_id, policy_type)

@app.function(image=image, gpu="any", timeout=86400, volumes={partition: vol}, secrets=[wandb_secret])
def train_policy(dataset_repo_id: str, policy_type: str):
    init_logging()

    policy_config = make_policy_config(policy_type)

    output_dir = get_output_dir(dataset_repo_id, policy_type)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_repo_id,
        ),
        policy=policy_config,
        output_dir=output_dir,
        wandb=WandBConfig(
            enable=True,
        )
    )
    train(cfg)

@app.function(image=image, volumes={partition: vol}, secrets=[hf_secret])
def upload_model(dataset_repo_id: str, policy_type: str):
    output_dir = get_output_dir(dataset_repo_id, policy_type)
    input_dir = output_dir.joinpath("checkpoints", "last", "pretrained_model")

    model_repo_id = f"{dataset_repo_id}_{policy_type}"

    upload_folder(
        folder_path=str(input_dir),
        repo_id=model_repo_id,
        repo_type="model"
    )