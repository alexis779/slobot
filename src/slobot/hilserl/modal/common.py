"""Shared Modal image, volume, and dataset setup for HIL-SERL scripts."""

from __future__ import annotations

import json
from pathlib import Path

import modal

REMOTE_REPO = "/root/slobot"
HF_CACHE_DIR = "/vol/huggingface"
HF_LEROBOT_HOME = f"{HF_CACHE_DIR}/lerobot"


def local_slobot_src() -> Path:
    """Path to src/slobot on the machine that builds the image (not inside Modal runtime)."""
    app_file = Path(__file__).resolve()
    try:
        return app_file.parents[4] / "src" / "slobot"
    except IndexError:
        # Modal imports this file as /root/common.py; source tree is already in the image.
        return Path(f"{REMOTE_REPO}/src/slobot")


def build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "libglib2.0-0", "libgl1", "ffmpeg")
        .env({"GIT_LFS_SKIP_SMUDGE": "1"})
        .pip_install(
            "torch",
            # PyPI lerobot lacks gaussian_actor (HIL-SERL policy); match doc/installation.md.
            "lerobot[hilserl,dataset,training,grpcio-dep] @ git+https://github.com/huggingface/lerobot.git",
            "transformers",
            "wandb",
            "draccus",
            "huggingface-hub>=1.5.0,<2.0",
            "gymnasium>=1.2.3",
            "opencv-python-headless>=4.13.0.92",
            "feetech-servo-sdk",
            "rerun-sdk",
        )
        .add_local_dir(
            local_slobot_src(),
            remote_path=f"{REMOTE_REPO}/src/slobot",
            copy=True,
        )
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": f"{REMOTE_REPO}/src",
                "HF_HOME": HF_CACHE_DIR,
                "HF_LEROBOT_HOME": HF_LEROBOT_HOME,
            }
        )
    )


image = build_image()
output_volume = modal.Volume.from_name("slobot-hilserl-outputs", create_if_missing=True)

MODAL_SECRETS = [
    modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"]),
    modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]),
]

def prefetch_offline_dataset(config_path: str) -> None:
    path = Path(config_path)
    cfg = json.loads(path.read_text())
    dataset = cfg.get("dataset") or {}
    repo_id = dataset.get("repo_id")
    if not repo_id:
        return
    from huggingface_hub import snapshot_download
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    print(f"[modal] Prefetching offline dataset {repo_id} into {HF_LEROBOT_HUB_CACHE} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=HF_LEROBOT_HUB_CACHE)
    print("[modal] Dataset prefetch done.")
