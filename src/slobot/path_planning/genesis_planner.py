"""
Genesis built-in path planning for rigid entities.

Uses Genesis entity.plan_path for joint-space motion planning. Supports batched
environments and runs the same policy across multiple simulation instances.
"""

from __future__ import annotations

import torch

from slobot.path_planning.path_interpolator import PathInterpolator


class GenesisPathPlanner:
    """
    Path planner using Genesis built-in plan_path.

    Plans from the entity's current configuration to a target configuration
    using Genesis's RRT/RRTConnect implementation.
    """

    def __init__(self, entity):
        """
        Parameters
        ----------
        entity : RigidEntity
            The entity to plan for (e.g. a robot arm).
        """
        self._entity = entity
        self._n_dofs = entity.n_dofs

    def plan(
        self,
        target_qpos: torch.Tensor | list | tuple,
    ) -> torch.Tensor:
        """
        Plan a path from current configuration to target using Genesis.

        Parameters
        ----------
        target_qpos : torch.Tensor
            Target joint configuration. Shape (n_dofs,) or (1, n_dofs).

        Returns
        -------
        path : torch.Tensor
            Path of shape (n_waypoints, 1, n_dofs) from current to target qpos.
        """
        path, valid_mask = self._entity.plan_path(
            qpos_goal=target_qpos,
            return_valid_mask=True,
            num_waypoints=PathInterpolator.NUM_WAYPOINTS,
        )
        if not valid_mask.all():
            raise ValueError("Path planning failed")
        return path
