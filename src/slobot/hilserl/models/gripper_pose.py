"""Gripper link pose in world frame."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperLinkPose:
    """End-effector pose of gripper_link: position (m) + rotation vector (rad)."""

    position: tuple[float, float, float]
    rotvec: tuple[float, float, float]
