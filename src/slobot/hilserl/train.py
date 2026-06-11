"""Train the HIL-SERL reward classifier from parquet camera frames recorded by slobot-record."""

from __future__ import annotations

# Register custom robot, teleop, and processor configs before LeRobot parses CLI args.
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config

register_hilserl_processor_config()

import lerobot.rewards  # noqa: F401 — register reward_classifier config subclass

from lerobot.datasets import factory as dataset_factory
from lerobot.scripts import lerobot_train

from slobot.hilserl.parquet_camera_dataset import (
    ensure_parquet_camera_features,
    make_parquet_camera_dataset,
)

_orig_make_dataset = dataset_factory.make_dataset


def _make_reward_classifier_dataset(cfg):
    ensure_parquet_camera_features(cfg)
    return make_parquet_camera_dataset(cfg)


def train_cli() -> None:
    lerobot_train.register_third_party_plugins()
    dataset_factory.make_dataset = _make_reward_classifier_dataset
    try:
        lerobot_train.main()
    finally:
        dataset_factory.make_dataset = _orig_make_dataset


def main() -> None:
    train_cli()


if __name__ == "__main__":
    main()
