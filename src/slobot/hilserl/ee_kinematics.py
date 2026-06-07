"""End-effector pose normalization for HIL-SERL (gripper_link xyz + rotvec + jaw)."""

from __future__ import annotations

import math

import numpy as np
import torch

EE_STATE_NAMES = ("x", "y", "z", "wx", "wy", "wz", "jaw")
EE_STATE_DIM = len(EE_STATE_NAMES)

# MIN_MAX bounds; jaw upper bound updated from sim limits at runtime when available
EE_STATE_MIN = [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi, -1.75]
EE_STATE_MAX = [1.0, 1.0, 1.0, math.pi, math.pi, math.pi, 1.75]
EE_STATE_EPS = 1e-8


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

def update_jaw_bounds(jaw_min: float, jaw_max: float) -> None:
    """Update global jaw normalization bounds from sim dof limits."""
    global EE_STATE_MIN, EE_STATE_MAX
    mins = list(EE_STATE_MIN)
    maxs = list(EE_STATE_MAX)
    mins[6] = jaw_min
    maxs[6] = jaw_max
    EE_STATE_MIN = mins
    EE_STATE_MAX = maxs
