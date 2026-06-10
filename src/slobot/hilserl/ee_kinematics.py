"""End-effector pose normalization for HIL-SERL (gripper_link xyz + 6D rotation + jaw)."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

EE_STATE_NAMES = (
    "x",
    "y",
    "z",
    "r1_x",
    "r1_y",
    "r1_z",
    "r2_x",
    "r2_y",
    "r2_z",
    "jaw",
)
EE_STATE_DIM = len(EE_STATE_NAMES)
ROTATION_6D_DIM = 6

# MIN_MAX bounds; rotation matrix columns and jaw use [-1, 1].
EE_STATE_MIN = [-1.0] * EE_STATE_DIM
EE_STATE_MAX = [1.0] * EE_STATE_DIM
EE_STATE_EPS = 1e-8


def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """First two columns of a 3x3 rotation matrix (Zhou et al.)."""
    r = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    return (
        float(r[0, 0]),
        float(r[1, 0]),
        float(r[2, 0]),
        float(r[0, 1]),
        float(r[1, 1]),
        float(r[2, 1]),
    )


def rotation_6d_to_matrix(rotation_6d: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Recover a 3x3 rotation matrix from a 6D rotation via Gram-Schmidt."""
    values = np.asarray(rotation_6d, dtype=float).reshape(ROTATION_6D_DIM)
    a1 = values[:3]
    a2 = values[3:6]

    norm_a1 = np.linalg.norm(a1)
    if norm_a1 < EE_STATE_EPS:
        a1 = np.array([1.0, 0.0, 0.0], dtype=float)
        norm_a1 = 1.0
    b1 = a1 / norm_a1

    a2_proj = a2 - np.dot(b1, a2) * b1
    norm_a2 = np.linalg.norm(a2_proj)
    if norm_a2 < EE_STATE_EPS:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(np.dot(b1, fallback)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        a2_proj = fallback - np.dot(b1, fallback) * b1
        norm_a2 = np.linalg.norm(a2_proj)
    b2 = a2_proj / norm_a2
    b3 = np.cross(b1, b2)
    return np.column_stack((b1, b2, b3))


def quat_wxyz_to_rotation_6d(quat_wxyz: list[float] | tuple[float, ...]) -> tuple[float, float, float, float, float, float]:
    rotation_matrix = ScipyRotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return rotation_matrix_to_6d(rotation_matrix)


def rotation_6d_to_quat_wxyz(
    rotation_6d: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[float, float, float, float]:
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)
    quat = ScipyRotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def clip_ee_xyz(
    x: float,
    y: float,
    z: float,
    end_effector_bounds: dict,
) -> tuple[float, float, float]:
    """Clip EE position (meters) to workspace bounds."""
    pos_min = np.asarray(end_effector_bounds["min"], dtype=float)
    pos_max = np.asarray(end_effector_bounds["max"], dtype=float)
    clipped = np.clip([x, y, z], pos_min, pos_max)
    return float(clipped[0]), float(clipped[1]), float(clipped[2])


def limit_ee_position_step(
    pos: np.ndarray,
    last_pos: np.ndarray | None,
    max_ee_step_m: float,
) -> np.ndarray:
    """Cap Euclidean EE motion between consecutive policy commands."""
    if last_pos is None:
        return pos
    dpos = pos - last_pos
    step = float(np.linalg.norm(dpos))
    if step > max_ee_step_m and step > 0:
        return last_pos + dpos * (max_ee_step_m / step)
    return pos


def normalize_ee_state(values: torch.Tensor | list[float] | np.ndarray) -> torch.Tensor:
    """Map physical EE state to [-1, 1] using MIN_MAX."""
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    min_vals = torch.tensor(EE_STATE_MIN, dtype=torch.float32)
    max_vals = torch.tensor(EE_STATE_MAX, dtype=torch.float32)
    denom = max_vals - min_vals
    denom = torch.where(denom == 0, torch.tensor(EE_STATE_EPS), denom)
    return 2.0 * (tensor - min_vals) / denom - 1.0


def denormalize_ee_state(values: torch.Tensor | list[float] | np.ndarray) -> torch.Tensor:
    """Invert MIN_MAX normalization back to physical EE state."""
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    min_vals = torch.tensor(EE_STATE_MIN, dtype=torch.float32)
    max_vals = torch.tensor(EE_STATE_MAX, dtype=torch.float32)
    return (tensor + 1.0) * 0.5 * (max_vals - min_vals) + min_vals
