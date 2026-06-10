"""Run the HIL-SERL learner on Modal with a public gRPC tunnel for a remote actor.

Usage:
  uv run modal run src/slobot/hilserl/modal/learner.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import modal

from slobot.hilserl.modal.common import (
    MODAL_SECRETS,
    REMOTE_REPO,
    image,
    output_volume,
    prefetch_offline_dataset,
)

REMOTE_CONFIG = f"{REMOTE_REPO}/src/slobot/hilserl/configs/train_hil_serl_config.json"
RESUME_OUTPUT_DIR = "output"
GRPC_PORT = 50051
VOLUME_ROOT = "/vol"

app = modal.App("slobot-hilserl-learner")


def resume_train_config_path(output_dir: str) -> str:
    return f"{output_dir}/checkpoints/last/pretrained_model/train_config.json"


def _ensure_last_checkpoint_link(output_dir: str) -> None:
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        return

    last = checkpoints_dir / "last"
    marker = last / "pretrained_model" / "train_config.json"
    if marker.is_file():
        return

    step_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not step_dirs:
        return

    latest = step_dirs[-1]
    if last.exists() or last.is_symlink():
        last.unlink(missing_ok=True)
    last.symlink_to(latest.name)


def _can_resume(output_dir: str) -> bool:
    """Return True when a checkpoint exists under output_dir on the Modal volume."""
    path = Path(output_dir)
    if not path.is_dir():
        return False

    _ensure_last_checkpoint_link(output_dir)
    return Path(resume_train_config_path(output_dir)).is_file()


def _prepare_fresh_output_dir(output_dir: str) -> None:
    """Remove an empty output tree so LeRobot can start with resume=false."""
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)


def _invoke_learner_cli(*, output_dir: str, config_path: str) -> None:
    # Register slobot plugins then run LeRobot learner (blocks forever).
    import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
    import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
    from lerobot.cameras import opencv  # noqa: F401
    from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config
    from slobot.hilserl.learner import train_cli

    register_hilserl_processor_config()
    # LeRobot defaults to outputs/train/{date}/{time}_{job_name}/ relative to cwd.
    os.chdir(VOLUME_ROOT)

    if _can_resume(output_dir):
        learner_config_path = resume_train_config_path(output_dir)
        resume = "true"
        print(f"Resuming training from {VOLUME_ROOT}/{learner_config_path}", flush=True)
    else:
        _prepare_fresh_output_dir(output_dir)
        learner_config_path = config_path
        resume = "false"
        print(
            f"No checkpoint under {VOLUME_ROOT}/{output_dir}; "
            f"starting fresh with {learner_config_path}",
            flush=True,
        )

    sys.argv = [
        "slobot-learner",
        f"--config_path={learner_config_path}",
        "--dataset.video_backend=pyav",
        "--num_workers=0",
        f"--resume={resume}",
        f"--output_dir={output_dir}",
        # Modal forwards external TCP to container :50051; gRPC must listen on all interfaces.
        "--policy.actor_learner_config.learner_host=0.0.0.0",
        f"--policy.actor_learner_config.learner_port={GRPC_PORT}",
    ]
    train_cli()


@app.function(
    image=image,
    gpu="any",
    timeout=24 * 60 * 60,
    volumes={"/vol": output_volume},
    secrets=MODAL_SECRETS,
)
def run_hilserl_learner(
    config_path: str = REMOTE_CONFIG,
    grpc_port: int = GRPC_PORT,
    output_dir: str = RESUME_OUTPUT_DIR,
) -> None:
    """Start learner gRPC server and expose it with an unencrypted TCP tunnel for the actor."""
    import torch

    prefetch_offline_dataset(config_path)

    bind_host = "0.0.0.0"
    with modal.forward(grpc_port, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print("=" * 72, flush=True)
        print("SLOBOT HIL-SERL LEARNER (Modal)", flush=True)
        print(f"  CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}", flush=True)
        print(f"  Output dir: {VOLUME_ROOT}/{output_dir}", flush=True)
        print(f"  gRPC bind (in container): {bind_host}:{grpc_port}", flush=True)
        print("  Connect the actor (computer) with:", flush=True)
        print(f"    --policy.actor_learner_config.learner_host={host}", flush=True)
        print(f"    --policy.actor_learner_config.learner_port={port}", flush=True)
        print("=" * 72, flush=True)

        _invoke_learner_cli(output_dir=output_dir, config_path=config_path)


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint for `modal run`."""
    print("Starting Modal learner. Watch the logs for the actor host:port to use on the computer.")
    print(f"Output dir on volume: {RESUME_OUTPUT_DIR}")
    run_hilserl_learner.remote(config_path=REMOTE_CONFIG)
