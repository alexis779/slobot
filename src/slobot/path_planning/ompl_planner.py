"""
OMPL-based path planning for Genesis rigid entities.

Plans collision-free paths in joint space from current configuration to a target
configuration using the Open Motion Planning Library (OMPL).

Installation options:
  - PyPI (Python 3.10-3.12): uv pip install ompl
  - Built from source (e.g. ~/Documents/python/robotics/ompl):
    cd ompl/build/Release && cmake ../.. -DOMPL_BUILD_PYTHON_BINDINGS=ON && make
    PYTHONPATH=ompl/build/Release/nanobinds:$PYTHONPATH python -m slobot.path_planning.ompl_planner

Usage:
    planner = OmplPathPlanner(entity, scene)
    path = planner.plan(target_qpos=goal)
"""

from __future__ import annotations

import torch

import genesis as gs

from slobot.path_planning.path_interpolator import PathInterpolator

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError as e:
    raise ImportError(
        "OMPL is required for this module. Install with: uv pip install ompl\n"
        "Note: OMPL supports Python 3.10-3.12; use a compatible Python version."
    ) from e

from slobot.configuration import Configuration

class OmplPathPlanner:
    """
    Path planner using OMPL for collision-free motion in joint space.

    Uses the scene's collision detection to validate states. Plans from the
    entity's current configuration to a target configuration.
    """

    LOGGER = Configuration.logger(__name__)

    def __init__(self, entity, scene):
        """
        Parameters
        ----------
        entity : RigidEntity
            The entity to plan for (e.g. a robot arm).
        scene : Scene
            The Genesis scene containing the entity. Used to access
            rigid_solver.collider for collision checking.
        """
        self._entity = entity
        self._scene = scene
        self._rigid_solver = scene.rigid_solver
        self._collider = self._rigid_solver.collider
        self._n_dofs = entity.n_dofs
        self._exclude_geom_pairs: set[tuple[int, int]] = set()
        self.interpolator = PathInterpolator()

    def _get_exclude_geom_pairs(self, qposs: list[torch.Tensor]) -> set[tuple[int, int]]:
        """Get geom pairs in contact at given configs (e.g. start/goal) to exclude from validity check."""
        pairs = set()
        for qpos in qposs:
            qpos = torch.as_tensor(qpos, device=gs.device).float().flatten()
            if self._rigid_solver.n_envs > 0:
                qpos = qpos.unsqueeze(0)
            self._entity.set_dofs_position(qpos, zero_velocity=False)
            self._rigid_solver._kernel_detect_collision()
            contact_info = self._entity.get_contacts()
            geom_a = contact_info["geom_a"]
            geom_b = contact_info["geom_b"]
            if geom_a.dim() > 1:
                geom_a, geom_b = geom_a[0], geom_b[0]
            valid = (geom_a >= 0) & (geom_b >= 0)
            for a, b in zip(geom_a[valid].tolist(), geom_b[valid].tolist()):
                pairs.add((int(a), int(b)))
                pairs.add((int(b), int(a)))  # symmetric
        return pairs

    def _is_state_valid(self, state: ob.RealVectorStateInternal) -> bool:
        """Check if a joint configuration is valid (collision-free, excluding allowed contacts)."""
        # Extract state as torch tensor
        qpos = torch.tensor(
            [state[i] for i in range(self._n_dofs)],
            dtype=torch.float32,
            device=gs.device,
        )
        # set_dofs_position expects (n_dofs,) for n_envs=0, (1, n_dofs) for batched
        if self._rigid_solver.n_envs > 0:
            qpos = qpos.unsqueeze(0)

        self._entity.set_dofs_position(qpos, zero_velocity=False)

        # Run collision detection
        self._rigid_solver._kernel_detect_collision()

        # Check for contacts, excluding pairs allowed at start/goal (e.g. base on table)
        contacts = self._scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
        geom_a = contacts.get("geom_a", torch.tensor([], device=gs.device))
        geom_b = contacts.get("geom_b", torch.tensor([], device=gs.device))
        if isinstance(geom_a, tuple):
            geom_a = geom_a[0] if geom_a else torch.tensor([], device=gs.device)
            geom_b = geom_b[0] if geom_b else torch.tensor([], device=gs.device)
        elif geom_a.dim() > 1:
            geom_a, geom_b = geom_a[0], geom_b[0]
        n_bad = 0
        for a, b in zip(geom_a.tolist(), geom_b.tolist()):
            if a >= 0 and b >= 0 and (a, b) not in self._exclude_geom_pairs and (b, a) not in self._exclude_geom_pairs:
                n_bad += 1
        return n_bad == 0

    def plan(
        self,
        target_qpos: torch.Tensor | list | tuple,
    ) -> torch.Tensor:
        """
        Plan a collision-free path from current configuration to target.

        Parameters
        ----------
        target_qpos : torch.Tensor
            Target joint configuration. Shape (n_dofs,) or (1, n_dofs).

        Returns
        -------
        path : torch.Tensor
            Path of shape (n_waypoints, 1, n_dofs) from current to target qpos.
        """
        # Ensure tensors on CPU for OMPL (Python floats)
        target_qpos = torch.as_tensor(target_qpos, device="cpu").float().flatten()
        target_qpos = target_qpos[: self._n_dofs]

        # Get current state and limits
        start_qpos0 = self._entity.get_dofs_position()
        start_qpos = torch.as_tensor(start_qpos0, device="cpu").float().flatten()
        if start_qpos.shape[0] != self._n_dofs:
            start_qpos = start_qpos.reshape(-1, self._n_dofs)[0]

        lower, upper = self._entity.get_dofs_limit()
        lower = torch.as_tensor(lower, device="cpu").float().flatten()
        upper = torch.as_tensor(upper, device="cpu").float().flatten()
        if lower.shape[0] != self._n_dofs:
            lower = lower.reshape(-1, self._n_dofs)[0]
        if upper.shape[0] != self._n_dofs:
            upper = upper.reshape(-1, self._n_dofs)[0]
        # Clamp start to limits (neutral config may exceed limits)
        start_qpos = torch.clamp(start_qpos, lower, upper)

        # Exclude geom pairs in contact at start/goal (e.g. base on table)
        self._exclude_geom_pairs = self._get_exclude_geom_pairs(
            [start_qpos.to(gs.device), target_qpos.to(gs.device)]
        )

        # Create OMPL state space
        space = ob.RealVectorStateSpace(self._n_dofs)
        bounds = ob.RealVectorBounds(self._n_dofs)
        for i in range(self._n_dofs):
            bounds.setLow(i, float(lower[i].item()))
            bounds.setHigh(i, float(upper[i].item()))
        space.setBounds(bounds)

        # SimpleSetup with state validity checker
        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(self._is_state_valid)

        # Set start and goal (OMPL 2.0: allocState + direct indexing)
        start = ss.getStateSpace().allocState()
        for i in range(self._n_dofs):
            start[i] = float(start_qpos[i].item())

        goal = ss.getStateSpace().allocState()
        for i in range(self._n_dofs):
            goal[i] = float(target_qpos[i].item())

        ss.setStartAndGoalStates(start, goal)

        # Use RRTConnect (more reliable for joint-space planning)
        ss.setPlanner(og.RRTConnect(ss.getSpaceInformation()))

        # Solve (time limit in seconds)
        solved = ss.solve(5.0)
        if not solved:
            raise RuntimeError("OMPL failed to find a path to the target configuration.")

        # Restore start configuration
        self._entity.set_dofs_position(start_qpos0, zero_velocity=False)

        # Extract raw OMPL path
        path_ompl = ss.getSolutionPath()
        n_waypoints = path_ompl.getStateCount()
        if n_waypoints < 2:
            path = torch.stack(
                [start_qpos.to(gs.device), target_qpos.to(gs.device)],
                dim=0,
            )
            return path.unsqueeze(1)  # (n_waypoints, 1, n_dofs)

        # Get waypoints from OMPL path (OMPL 2.0: state[j] direct indexing)
        waypoints = torch.zeros(n_waypoints, self._n_dofs, dtype=torch.float32, device=gs.device)
        for i in range(n_waypoints):
            s = path_ompl.getState(i)
            for j in range(self._n_dofs):
                waypoints[i, j] = s[j]

        path = waypoints.unsqueeze(1)  # (n_waypoints, 1, n_dofs)
        self.LOGGER.info(f"path = {path}")
        path = self.interpolator.interpolate(path)
        return path