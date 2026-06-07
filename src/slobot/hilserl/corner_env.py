"""Genesis corner walls for HIL-SERL workspace limits."""

from __future__ import annotations

import genesis as gs

from slobot.robotic_arm import RoboticArm


class CornerEnv:
    """Two vertical planes at the tips of the workspace boundary arrows."""

    # Arrow tip position (m) and plane normal (unit, world frame).
    WALL_SPECS = (
        ((-0.29, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 0.1, 0.0), (0.0, 1.0, 0.0)),
    )

    def __init__(self, robotic_arm: RoboticArm) -> None:
        self.arm = robotic_arm
        self.walls: list = []
        self.build_scene()

    def build_scene(self) -> None:
        genesis = self.arm.genesis
        genesis.start()

        vis_mode = genesis.vis_mode
        for pos, normal in self.WALL_SPECS:
            wall = genesis.scene.add_entity(
                gs.morphs.Plane(pos=pos, normal=normal, fixed=True),
                visualize_contact=False,
                vis_mode=vis_mode,
            )
            self.walls.append(wall)

        genesis.build()
