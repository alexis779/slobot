"""Packed EE observation and action for policy / dataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slobot.hilserl.ee_kinematics import EE_STATE_DIM, denormalize_ee_state, normalize_ee_state
from slobot.hilserl.models.gripper_pose import GripperLinkPose


@dataclass(frozen=True)
class EeObservation:
    pose: GripperLinkPose
    jaw_rad: float

    def to_tensor(self) -> torch.Tensor:
        values = [
            *self.pose.position,
            *self.pose.rotation_6d,
            self.jaw_rad,
        ]
        return normalize_ee_state(values)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor | list[float]) -> EeObservation:
        physical = denormalize_ee_state(tensor)
        vals = [float(v) for v in physical.reshape(-1).tolist()]
        if len(vals) != EE_STATE_DIM:
            raise ValueError(f"Expected {EE_STATE_DIM} values, got {len(vals)}")
        return cls(
            pose=GripperLinkPose(
                position=(vals[0], vals[1], vals[2]),
                rotation_6d=(vals[3], vals[4], vals[5], vals[6], vals[7], vals[8]),
            ),
            jaw_rad=vals[9],
        )


@dataclass(frozen=True)
class EeAction:
    pose: GripperLinkPose
    jaw_rad: float

    def to_tensor(self) -> torch.Tensor:
        values = [
            *self.pose.position,
            *self.pose.rotation_6d,
            self.jaw_rad,
        ]
        return normalize_ee_state(values)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor | list[float]) -> EeAction:
        physical = denormalize_ee_state(tensor)
        vals = [float(v) for v in physical.reshape(-1).tolist()]
        if len(vals) != EE_STATE_DIM:
            raise ValueError(f"Expected {EE_STATE_DIM} values, got {len(vals)}")
        return cls(
            pose=GripperLinkPose(
                position=(vals[0], vals[1], vals[2]),
                rotation_6d=(vals[3], vals[4], vals[5], vals[6], vals[7], vals[8]),
            ),
            jaw_rad=vals[9],
        )
