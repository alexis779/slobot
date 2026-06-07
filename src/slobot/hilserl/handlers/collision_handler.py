"""Collision checking via Genesis entity contact API."""

from __future__ import annotations

import genesis as gs

from slobot.hilserl.handlers.motor_qpos import motor_to_sim_qpos
from slobot.hilserl.models.collision import CollisionResult
from slobot.hilserl.models.motor_io import MotorRadians
from slobot.robotic_arm import RoboticArm


class GenesisCollisionHandler:
    def __init__(self, robotic_arm: RoboticArm, *, n_dofs: int):
        self._entity = robotic_arm.genesis.entity
        self._n_dofs = n_dofs
        self._scene = robotic_arm.genesis.scene
        self._rigid_solver = self._scene.rigid_solver
        self._exclude_geom_pairs: set[tuple[int, int]] = set()
        self._capture_baseline_contacts()

    def _capture_baseline_contacts(self) -> None:
        """Record geom pairs in contact at the current (home) configuration."""
        self._rigid_solver._kernel_detect_collision()
        contact_info = self._entity.get_contacts()
        geom_a = contact_info["geom_a"]
        geom_b = contact_info["geom_b"]
        if geom_a.dim() > 1:
            geom_a, geom_b = geom_a[0], geom_b[0]
        valid = (geom_a >= 0) & (geom_b >= 0)
        for a, b in zip(geom_a[valid].tolist(), geom_b[valid].tolist()):
            self._exclude_geom_pairs.add((int(a), int(b)))
            self._exclude_geom_pairs.add((int(b), int(a)))

    def check_collision(self, radians: MotorRadians) -> CollisionResult:
        saved_qpos = self._entity.get_qpos().clone()

        qpos = motor_to_sim_qpos(radians, n_dofs=self._n_dofs).to(device=gs.device)
        if self._rigid_solver.n_envs > 0:
            qpos = qpos.unsqueeze(0)
        self._entity.set_dofs_position(qpos, zero_velocity=False)
        self._rigid_solver._kernel_detect_collision()

        contact_info = self._entity.get_contacts()
        geom_a = contact_info["geom_a"]
        geom_b = contact_info["geom_b"]
        if geom_a.dim() > 1:
            geom_a, geom_b = geom_a[0], geom_b[0]

        n_bad = 0
        valid = (geom_a >= 0) & (geom_b >= 0)
        for a, b in zip(geom_a[valid].tolist(), geom_b[valid].tolist()):
            a_int, b_int = int(a), int(b)
            if (a_int, b_int) not in self._exclude_geom_pairs and (b_int, a_int) not in self._exclude_geom_pairs:
                n_bad += 1

        self._entity.set_dofs_position(saved_qpos, zero_velocity=False)
        return CollisionResult(valid=n_bad == 0, contact_count=n_bad)
