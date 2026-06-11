"""Train the HIL-SERL reward classifier on Modal.

Usage:
  uv run modal run src/slobot/hilserl/modal/train_reward_classifier.py

Optional overrides (passed to the remote function):
  uv run modal run src/slobot/hilserl/modal/train_reward_classifier.py --steps 50
"""

from __future__ import annotations

import os
import sys

import modal

from slobot.hilserl.modal.common import (
    MODAL_SECRETS,
    REMOTE_REPO,
    image,
    output_volume,
    prefetch_offline_dataset,
)

REMOTE_CONFIG = f"{REMOTE_REPO}/src/slobot/hilserl/configs/train_reward_classifier_config.json"
VOLUME_ROOT = "/vol"

app = modal.App("slobot-reward-classifier")


def _invoke_reward_classifier_train_cli(
    *,
    config_path: str,
    steps: int | None = None,
) -> None:
    from slobot.hilserl.train import train_cli

    # LeRobot defaults to outputs/train/{date}/{time}_{job_name}/ relative to cwd.
    os.chdir(VOLUME_ROOT)

    argv = [
        "slobot-train",
        "--config_path",
        config_path,
        "--reward_model.device=cpu",
        "--num_workers=0",
    ]
    if steps is not None:
        argv.append(f"--steps={steps}")
    sys.argv = argv
    train_cli()


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    volumes={"/vol": output_volume},
    secrets=MODAL_SECRETS,
)
def run_reward_classifier_training(
    config_path: str = REMOTE_CONFIG,
    steps: int | None = None,
) -> None:
    """Train the reward classifier on CPU and write checkpoints to the Modal volume."""
    prefetch_offline_dataset(config_path)

    print("=" * 72, flush=True)
    print("SLOBOT REWARD CLASSIFIER TRAINING (Modal, CPU)", flush=True)
    print(f"  Config: {config_path}", flush=True)
    print(f"  Checkpoints / logs: {VOLUME_ROOT}/outputs/train/{{date}}/{{time}}_reward-classifier/", flush=True)
    if steps is not None:
        print(f"  Steps override: {steps}", flush=True)
    print("=" * 72, flush=True)

    _invoke_reward_classifier_train_cli(
        config_path=config_path,
        steps=steps,
    )


@app.local_entrypoint()
def main(steps: int = 0) -> None:
    """Local entrypoint for `modal run`. Pass --steps N to override training length."""
    kwargs: dict = {"config_path": REMOTE_CONFIG}
    if steps > 0:
        kwargs["steps"] = steps
    print("Starting Modal reward classifier training.")
    run_reward_classifier_training.remote(**kwargs)
