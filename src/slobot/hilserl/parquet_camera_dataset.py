"""Ensure reward-classifier training reads live preprocessed tensors from parquet."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from lerobot.processor.rename_processor import rename_stats

from slobot.hilserl.dataset_image_keys import (
    default_preprocessed_to_video_rename_map,
    is_parquet_preprocessed_key,
    preprocessed_key_for_video_key,
)

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)


def _dataset_root(cfg: TrainPipelineConfig) -> Path:
    if cfg.dataset.root is not None:
        return Path(cfg.dataset.root)
    return HF_LEROBOT_HOME / cfg.dataset.repo_id


def _parquet_camera_feature(shape: tuple[int, ...]) -> dict:
    return {
        "dtype": "float32",
        "shape": list(shape),
        "names": ["channels", "height", "width"],
    }


def parquet_preprocessed_keys(meta: LeRobotDatasetMetadata) -> list[str]:
    return [
        key
        for key, feature in meta.features.items()
        if is_parquet_preprocessed_key(key) and feature.get("dtype") == "float32"
    ]


def _dataset_has_preprocessed_parquet(meta: LeRobotDatasetMetadata) -> bool:
    return len(parquet_preprocessed_keys(meta)) > 0


def _preprocessed_to_classifier_rename_map(meta: LeRobotDatasetMetadata) -> dict[str, str]:
    return default_preprocessed_to_video_rename_map(parquet_preprocessed_keys(meta))


def _rename_dataset_item(item: dict, rename_map: dict[str, str]) -> dict:
    renamed = dict(item)
    for src, dst in rename_map.items():
        if src in renamed:
            renamed[dst] = renamed.pop(src)
    return renamed


class _RenamingLeRobotDataset:
    """Rename parquet preprocessed keys to classifier input keys at sample time."""

    def __init__(self, dataset: LeRobotDataset, rename_map: dict[str, str]):
        self._dataset = dataset
        self._rename_map = rename_map
        if dataset.meta.stats is not None:
            dataset.meta.stats = rename_stats(dataset.meta.stats, rename_map)

    def __getitem__(self, idx: int) -> dict:
        return _rename_dataset_item(self._dataset[idx], self._rename_map)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)


def ensure_parquet_camera_features(cfg: TrainPipelineConfig) -> None:
    """Backfill observation.preprocessed.* from AV1 video when missing (legacy datasets)."""
    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    if _dataset_has_preprocessed_parquet(meta):
        return

    video_keys = meta.video_keys
    if not video_keys:
        raise ValueError(
            f"Dataset {cfg.dataset.repo_id} has no observation.preprocessed.* parquet columns "
            "and no observation.images.* videos to migrate from. Re-record with slobot-record."
        )

    logger.info(
        "Dataset %s lacks preprocessed parquet cameras; backfilling from AV1 video decode.",
        cfg.dataset.repo_id,
    )

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=False,
    )
    num_frames = len(dataset)
    preprocessed_keys = [preprocessed_key_for_video_key(key) for key in video_keys]
    images_by_key: dict[str, list[np.ndarray]] = {
        key: [] for key in preprocessed_keys if key is not None
    }

    for idx in range(num_frames):
        item = dataset[idx]
        for video_key, pq_key in zip(video_keys, preprocessed_keys, strict=True):
            if pq_key is None:
                continue
            tensor = item[video_key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            images_by_key[pq_key].append(tensor.detach().cpu().numpy().astype(np.float32))

    root = _dataset_root(cfg)
    data_dir = root / "data"
    parquet_paths = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet data files under {data_dir}")

    frame_offset = 0
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        num_rows = table.num_rows
        for video_key, pq_key in zip(video_keys, preprocessed_keys, strict=True):
            if pq_key is None or pq_key in table.column_names:
                continue
            channels, height, width = (int(v) for v in meta.features[video_key]["shape"])
            column = pa.array(
                [
                    images_by_key[pq_key][frame_offset + row_idx].tolist()
                    for row_idx in range(num_rows)
                ],
                type=pa.list_(
                    pa.list_(pa.list_(pa.float32(), width), height),
                    channels,
                ),
            )
            table = table.append_column(pq_key, column)
        pq.write_table(table, parquet_path)
        frame_offset += num_rows

    if frame_offset != num_frames:
        raise RuntimeError(
            f"Parquet migration row count mismatch: wrote {frame_offset} rows, expected {num_frames}"
        )

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for video_key, pq_key in zip(video_keys, preprocessed_keys, strict=True):
        if pq_key is None or pq_key in info["features"]:
            continue
        camera = video_key.split("observation.images.", 1)[1]
        shape = tuple(meta.features[video_key]["shape"])
        info["features"][pq_key] = _parquet_camera_feature(shape)
        logger.info("Added parquet feature %s (from %s)", pq_key, camera)
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def make_parquet_camera_dataset(cfg: TrainPipelineConfig):
    """Load dataset; training reads observation.preprocessed.* (renamed to observation.images.*)."""
    from lerobot.datasets.factory import make_dataset

    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    if not _dataset_has_preprocessed_parquet(meta):
        raise ValueError(
            f"Dataset {cfg.dataset.repo_id} has no observation.preprocessed.* parquet columns. "
            "Re-record with slobot-record or run migration."
        )
    rename_map = _preprocessed_to_classifier_rename_map(meta)
    dataset = make_dataset(cfg)
    if not rename_map:
        return dataset
    return _RenamingLeRobotDataset(dataset, rename_map)
