"""Gripper link pose in world frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperLinkPose:
    """End-effector pose of gripper_link: position (m) + 6D rotation (Zhou et al.)."""

    position: tuple[float, float, float]
    rotation_6d: tuple[float, float, float, float, float, float]
