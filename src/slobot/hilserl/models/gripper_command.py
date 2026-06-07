"""Discrete gripper commands for HIL-SERL actions."""

from __future__ import annotations

from enum import Enum


class GripperCommand(Enum):
    OPEN = "open"
    STAY = "stay"
    CLOSE = "close"

    @classmethod
    def from_normalized(cls, value: float) -> GripperCommand:
        if value < -0.33:
            return cls.CLOSE
        if value > 0.33:
            return cls.OPEN
        return cls.STAY

    def to_normalized(self) -> float:
        if self is GripperCommand.OPEN:
            return 1.0
        if self is GripperCommand.CLOSE:
            return -1.0
        return 0.0
