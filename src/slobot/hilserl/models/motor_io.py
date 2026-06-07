"""Immutable motor position containers for HIL-SERL handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class MotorSteps:
    """Raw Feetech motor step positions (one per joint)."""

    tensor: torch.Tensor

    @classmethod
    def from_list(cls, values: Sequence[int | float]) -> MotorSteps:
        return cls(torch.tensor([int(v) for v in values], dtype=torch.int32))

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor | Sequence[int | float]) -> MotorSteps:
        if isinstance(tensor, MotorSteps):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return cls(tensor.reshape(-1).to(dtype=torch.int32))
        return cls.from_list(tensor)

    def to_tensor(self) -> torch.Tensor:
        return self.tensor

    def to_list(self) -> list[int]:
        return self.tensor.reshape(-1).tolist()

    def __len__(self) -> int:
        return self.tensor.numel()


@dataclass(frozen=True)
class MotorRadians:
    """Joint positions in radians (follower/leader motor bus order)."""

    tensor: torch.Tensor

    @classmethod
    def from_list(cls, values: Sequence[float]) -> MotorRadians:
        return cls(torch.tensor([float(v) for v in values], dtype=torch.float32))

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor | Sequence[float]) -> MotorRadians:
        if isinstance(tensor, MotorRadians):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return cls(tensor.reshape(-1).to(dtype=torch.float32))
        return cls.from_list(tensor)

    @classmethod
    def zeros(cls, n: int) -> MotorRadians:
        return cls(torch.zeros(n, dtype=torch.float32))

    def to_tensor(self) -> torch.Tensor:
        return self.tensor

    def to_list(self) -> list[float]:
        return self.tensor.reshape(-1).tolist()

    def __len__(self) -> int:
        return self.tensor.numel()

    def at(self, motor_idx: int) -> float:
        return float(self.tensor[motor_idx].item())
