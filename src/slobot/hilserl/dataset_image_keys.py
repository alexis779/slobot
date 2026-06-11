"""Dataset column naming for video vs live preprocessed camera tensors."""

from __future__ import annotations

OBS_IMAGES_VIDEO_PREFIX = "observation.images"
OBS_PREPROCESSED_PREFIX = "observation.preprocessed"
# Older recordings used this prefix; still loaded for training migration.
LEGACY_PREPROCESSED_IMAGES_PREFIX = "observation.preprocessed_images"


def video_image_key(camera: str) -> str:
    return f"{OBS_IMAGES_VIDEO_PREFIX}.{camera}"


def preprocessed_camera_key(camera: str) -> str:
    """Parquet column for live-resized float32 [C, H, W] tensors (classifier training)."""
    return f"{OBS_PREPROCESSED_PREFIX}.{camera}"


def is_video_image_key(key: str) -> bool:
    return key.startswith(f"{OBS_IMAGES_VIDEO_PREFIX}.")


def is_preprocessed_camera_key(key: str) -> bool:
    return key.startswith(f"{OBS_PREPROCESSED_PREFIX}.")


def is_parquet_preprocessed_key(key: str) -> bool:
    """Parquet preprocessed camera column (current or legacy naming)."""
    return is_preprocessed_camera_key(key) or key.startswith(
        f"{LEGACY_PREPROCESSED_IMAGES_PREFIX}."
    )


def camera_name_from_video_key(key: str) -> str | None:
    prefix = f"{OBS_IMAGES_VIDEO_PREFIX}."
    if not key.startswith(prefix):
        return None
    return key[len(prefix) :]


def camera_name_from_preprocessed_key(key: str) -> str | None:
    for prefix in (f"{OBS_PREPROCESSED_PREFIX}.", f"{LEGACY_PREPROCESSED_IMAGES_PREFIX}."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def preprocessed_key_for_video_key(video_key: str) -> str | None:
    camera = camera_name_from_video_key(video_key)
    if camera is None:
        return None
    return preprocessed_camera_key(camera)


def classifier_key_for_preprocessed_key(preprocessed_key: str) -> str | None:
    camera = camera_name_from_preprocessed_key(preprocessed_key)
    if camera is None:
        return None
    return video_image_key(camera)


def default_preprocessed_to_video_rename_map(preprocessed_keys: list[str]) -> dict[str, str]:
    """Map parquet preprocessed columns to classifier input keys for training."""
    rename_map: dict[str, str] = {}
    for preprocessed_key in preprocessed_keys:
        classifier_key = classifier_key_for_preprocessed_key(preprocessed_key)
        if classifier_key is not None:
            rename_map[preprocessed_key] = classifier_key
    return rename_map
