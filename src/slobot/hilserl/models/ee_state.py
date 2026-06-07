"""Packed EE observation and action for policy / dataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slobot.hilserl.ee_kinematics import EE_STATE_DIM, denormalize_ee_state, normalize_ee_state
from slobot.hilserl.models.gripper_command import GripperCommand
from slobot.hilserl.models.gripper_pose import GripperLinkPose


@dataclass(frozen=True)
class EeObservation:
    pose: GripperLinkPose
    jaw_rad: float

    def to_tensor(self) -> torch.Tensor:
        values = [
            *self.pose.position,
            *self.pose.rotvec,
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
                rotvec=(vals[3], vals[4], vals[5]),
            ),
            jaw_rad=vals[6],
        )


@dataclass(frozen=True)
class EeAction:
    pose: GripperLinkPose
    command: GripperCommand

    def to_tensor(self) -> torch.Tensor:
        values = [
            *self.pose.position,
            *self.pose.rotvec,
            self.command.to_normalized(),
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
                rotvec=(vals[3], vals[4], vals[5]),
            ),
            command=GripperCommand.from_normalized(vals[6]),
        )
