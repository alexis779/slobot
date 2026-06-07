"""Shared FK/IK/collision handlers for HIL-SERL processors."""

from __future__ import annotations

from dataclasses import dataclass

from slobot.hilserl.handlers.collision_handler import GenesisCollisionHandler
from slobot.hilserl.handlers.fk_handler import FkHandler
from slobot.hilserl.handlers.ik_handler import IkHandler


@dataclass
class KinematicsBundle:
    fk: FkHandler
    ik: IkHandler
    collision: GenesisCollisionHandler
    n_dofs: int
    n_motors: int
    jaw_joint_idx: int
    jaw_motor_idx: int
    jaw_limits: tuple[float, float]
    gripper_link_name: str
    jaw_joint_name: str
