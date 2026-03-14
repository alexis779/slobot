"""
Path interpolation for joint-space trajectories.

Used by OmplPathPlanner and GenesisPathPlanner via composition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PathInterpolator:
    NUM_WAYPOINTS = 50

    """
    Resamples joint-space paths to a requested number of waypoints.

    No collision checking is performed.
    """

    def interpolate(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resample an existing joint-space path to the requested number of waypoints.

        No collision checking is performed.

        Parameters
        ----------
        x : torch.Tensor
            Path tensor of shape (n_waypoints, 1, n_dofs) (Genesis plan_path format).
        path_length : int
            Number of waypoints in the output path.

        Returns
        -------
        path : torch.Tensor
            Path of shape (path_length, n_dofs).
        """
        n_waypoints, _, n_dofs = x.shape
        path_length = self.NUM_WAYPOINTS
        if n_waypoints == path_length:
            return x.squeeze(1)
        if n_waypoints == 1:
            return x.squeeze(1).repeat(path_length, 1)

        # F.interpolate expects (N, C, L), where C is n_dofs and L is n_waypoints.
        x = x.permute(1, 2, 0)  # (1, n_dofs, n_waypoints)
        path = F.interpolate(x, size=path_length, mode="linear", align_corners=True)
        return path.squeeze(0).T  # (path_length, n_dofs)
