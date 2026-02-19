import torch
import genesis as gs
from importlib.resources import files

from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration


class GolfBallEnv:
    def __init__(self):
        # Create SoArm100 instance
        self.arm = SoArm100(
            should_start=False,
            rgb=True,
        )
        
        # Build scene with objects
        self.build_scene()

    def build_scene(self):
        self.arm.genesis.start()

        golf_ball_morph = gs.morphs.Sphere(
            radius=Configuration.GOLF_BALL_RADIUS,
        )
        self.golf_ball = self.arm.genesis.scene.add_entity(
            golf_ball_morph,
            visualize_contact=False, # True
            vis_mode='visual', # collision
        )

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup_morph = gs.morphs.Mesh(
            file=cup_filename,
        )
        self.cup = self.arm.genesis.scene.add_entity(cup_morph)

        n_envs = 1
        self.arm.genesis.build(n_envs=n_envs)

    def set_object_initial_positions(self, ball_x: float, ball_y: float, cup_x: float, cup_y: float):
        ball_x = ball_x * Configuration.INCHES_TO_METERS
        ball_y = ball_y * Configuration.INCHES_TO_METERS
        cup_x = cup_x * Configuration.INCHES_TO_METERS
        cup_y = cup_y * Configuration.INCHES_TO_METERS

        self.ball_pos = torch.tensor([[
            ball_x,
            ball_y,
            Configuration.GOLF_BALL_RADIUS
        ]])
        self.cup_pos = torch.tensor([[
            cup_x,
            cup_y,
            0.0
        ]])
        self.golf_ball.set_pos(self.ball_pos)
        self.cup.set_pos(self.cup_pos)

    def is_golf_ball_in_cup(self) -> bool:
        """
        Check if the golf ball is near the cup position.
        
        Returns:
            True if ball is within distance threshold of cup, False otherwise
        """
        ball_pos = self.golf_ball.get_pos()[0]
        cup_pos = self.cup.get_pos()[0]

        # Check distance in XY plane
        diff = ball_pos[:2] - cup_pos[:2]
        distance = torch.norm(diff)

        return distance < Configuration.DISTANCE_THRESHOLD
