"""Resolve discrete gripper commands to jaw joint radians."""

from __future__ import annotations

from slobot.feetech import Feetech
from slobot.hilserl.models.gripper_command import GripperCommand
from slobot.hilserl.models.motor_io import MotorRadians


def resolve_gripper_command(
    command: GripperCommand,
    jaw_limits: tuple[float, float],
    *,
    feetech: Feetech,
    jaw_motor_idx: int,
    q_guess: MotorRadians | None = None,
    current_jaw_rad: float | None = None,
) -> float:
    """Map OPEN/CLOSE/STAY to a jaw motor qpos (rad) via Feetech qpos_to_pos."""
    lower, upper = jaw_limits
    midpoint = (lower + upper) / 2.0
    if command is GripperCommand.STAY:
        if current_jaw_rad is not None:
            return current_jaw_rad
        if q_guess is not None:
            return q_guess.at(jaw_motor_idx)
        jaw_qpos = midpoint
    elif command is GripperCommand.OPEN:
        jaw_qpos = midpoint
    elif command is GripperCommand.CLOSE:
        jaw_qpos = lower
    else:
        jaw_qpos = midpoint

    return _snap_jaw_qpos_to_motor_radians(
        feetech, jaw_motor_idx, jaw_qpos, q_guess=q_guess
    )


def _snap_jaw_qpos_to_motor_radians(
    feetech: Feetech,
    jaw_motor_idx: int,
    jaw_qpos: float,
    *,
    q_guess: MotorRadians | None = None,
) -> float:
    """Convert a jaw qpos target to achievable motor-bus radians (qpos_to_pos roundtrip)."""
    if q_guess is not None:
        qpos = q_guess.to_list()
    else:
        qpos = feetech.get_qpos()
    qpos[jaw_motor_idx] = jaw_qpos
    pos = feetech.qpos_to_pos(qpos)
    return feetech.pos_to_qpos(pos)[jaw_motor_idx]


def resolve_gripper_motor_step(
    command: GripperCommand,
    jaw_limits: tuple[float, float],
    *,
    feetech: Feetech,
    jaw_motor_idx: int,
    q_guess: MotorRadians | None = None,
    current_jaw_rad: float | None = None,
) -> int:
    """Map OPEN/CLOSE/STAY to a follower gripper motor step."""
    jaw_rad = resolve_gripper_command(
        command,
        jaw_limits,
        feetech=feetech,
        jaw_motor_idx=jaw_motor_idx,
        q_guess=q_guess,
        current_jaw_rad=current_jaw_rad,
    )
    qpos = feetech.pos_to_qpos(feetech.get_pos())
    qpos[jaw_motor_idx] = jaw_rad
    return int(feetech.qpos_to_pos(qpos)[jaw_motor_idx])
