"""Genesis IK settings for HIL-SERL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlobotGenesisConfig:
    """Simulator settings."""

    jaw_joint_name: str
    gripper_link_name: str
    end_effector_step_sizes: dict[str, float] | None = None
