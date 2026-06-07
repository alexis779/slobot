"""Motor bus radians ↔ Genesis sim qpos helpers."""

from __future__ import annotations

import torch

from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.hilserl.models.gripper_command import GripperCommand
from slobot.hilserl.models.motor_io import MotorRadians
from slobot.robotic_arm import RoboticArm

GRIPPER_MOTOR_IDX = Configuration.JOINT_NAMES.index("gripper")


def jaw_joint_index(robotic_arm: RoboticArm, jaw_joint_name: str) -> int:
    """Entity joint index in ``[0, n_dofs - 1]`` for the jaw DOF."""
    return robotic_arm.genesis.entity.get_joint(jaw_joint_name).idx_local


def resolve_jaw_indices(
    robotic_arm: RoboticArm,
    jaw_joint_name: str,
    motor_names: list[str],
) -> tuple[int, int]:
    """Return ``(jaw_joint_idx, jaw_motor_idx)``."""
    jaw_joint_idx = jaw_joint_index(robotic_arm, jaw_joint_name)
    jaw_motor_idx = GRIPPER_MOTOR_IDX
    gripper_name = Configuration.JOINT_NAMES[jaw_motor_idx]
    if motor_names[jaw_motor_idx] != gripper_name:
        raise ValueError(
            f"Jaw motor index {jaw_motor_idx} ({motor_names[jaw_motor_idx]!r}) != "
            f"expected {gripper_name!r}; check MJCF joint order vs motor bus"
        )
    return jaw_joint_idx, jaw_motor_idx


def jaw_joint_limits(robotic_arm: RoboticArm, jaw_joint_name: str) -> tuple[float, float]:
    jaw_joint = robotic_arm.genesis.entity.get_joint(jaw_joint_name)
    lower, upper = jaw_joint.dofs_limit[0]
    return float(lower), float(upper)


def motor_to_sim_qpos(motor_rad: MotorRadians, *, n_dofs: int) -> torch.Tensor:
    values = motor_rad.to_list()[:n_dofs]
    sim_qpos = torch.zeros(n_dofs, dtype=torch.float32)
    for joint_idx, value in enumerate(values):
        sim_qpos[joint_idx] = value
    return sim_qpos


def sim_to_motor_radians(
    sim_qpos: torch.Tensor,
    *,
    n_motors: int,
    q_guess: MotorRadians | None = None,
) -> MotorRadians:
    flat = sim_qpos.reshape(-1)
    values: list[float] = []
    for motor_idx in range(n_motors):
        if motor_idx < flat.numel():
            values.append(float(flat[motor_idx].item()))
        elif q_guess is not None:
            values.append(q_guess.at(motor_idx))
        else:
            values.append(0.0)
    return MotorRadians.from_list(values)


def apply_gripper_to_joint_targets(
    joint_targets: dict[str, int],
    *,
    command: GripperCommand,
    jaw_limits: tuple[float, float],
    feetech: Feetech,
    jaw_motor_idx: int,
    motor_names: list[str],
    previous_gripper_step: float | None,
) -> None:
    """Override follower gripper motor step from teleop gripper command."""
    from slobot.hilserl.handlers.gripper_command_handler import resolve_gripper_motor_step

    gripper_key = f"{motor_names[jaw_motor_idx]}.pos"
    if command is GripperCommand.STAY:
        if previous_gripper_step is not None:
            joint_targets[gripper_key] = int(previous_gripper_step)
        return

    previous_jaw_rad = None
    if previous_gripper_step is not None:
        previous_jaw_rad = jaw_rad_from_motor_step(feetech, previous_gripper_step)
    joint_targets[gripper_key] = resolve_gripper_motor_step(
        command,
        jaw_limits,
        feetech=feetech,
        jaw_motor_idx=jaw_motor_idx,
        current_jaw_rad=previous_jaw_rad,
    )


def jaw_rad_from_motor_step(feetech: Feetech, jaw_motor_step: float) -> float:
    pos = feetech.get_pos()
    pos[GRIPPER_MOTOR_IDX] = int(jaw_motor_step)
    return feetech.pos_to_qpos(pos)[GRIPPER_MOTOR_IDX]
