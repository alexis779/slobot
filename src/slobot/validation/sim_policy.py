import torch
from enum import Enum
from importlib.resources import files

import genesis as gs
import genesis.utils.geom as gu

from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer


class PreGraspMode(Enum):
    VERTICAL = "vertical"
    VERTICAL_FLIP = "vertical-flip"
    HORIZONTAL = "horizontal"


class SimPolicy:
    """Policy to pick up a golf ball and place it in a cup using IK with wrist orientation constraint."""
    
    LOGGER = Configuration.logger(__name__)    

    CUP_Z = 6 # inches

    GRIPPER_OPENED_QPOS = 1.0
    GRIPPER_GRASPING_QPOS = 0.25
    GRIPPER_CLOSED_QPOS = -1.0

    def __init__(self, **kwargs):
        """
        Initialize the policy.
        
        Args:
            ball_x: Golf ball X position in inches
            ball_y: Golf ball Y position in inches
            cup_x: Cup X position in inches
            cup_y: Cup Y position in inches
        """
        # Extract parameters
        ball_x = kwargs['ball_x']
        ball_y = kwargs['ball_y']
        cup_x = kwargs['cup_x']
        cup_y = kwargs['cup_y']

        # Convert positions from inches to meters
        self.ball_pos = torch.tensor([
            ball_x * Configuration.INCHES_TO_METERS,
            ball_y * Configuration.INCHES_TO_METERS,
            Configuration.GOLF_BALL_RADIUS
        ])
        
        self.cup_pos = torch.tensor([
            cup_x * Configuration.INCHES_TO_METERS,
            cup_y * Configuration.INCHES_TO_METERS,
            0.0
        ])

        self.pre_grasp_mode = kwargs['pre_grasp_mode']

        # Create SoArm100 instance
        self.arm = SoArm100(
            should_start=False,
        )
        
        # Build scene with objects
        self.build_scene()

    def build_scene(self):
        self.arm.genesis.start()

        golf_ball_morph = gs.morphs.Sphere(
            radius=Configuration.GOLF_BALL_RADIUS,
            pos=self.ball_pos,
        )
        self.golf_ball = self.arm.genesis.scene.add_entity(
            golf_ball_morph,
            visualize_contact=False, # True
            vis_mode='visual', # collision
        )

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup_morph = gs.morphs.Mesh(
            file=cup_filename,
            pos=self.cup_pos
        )
        self.cup = self.arm.genesis.scene.add_entity(cup_morph)

        n_envs = 1
        self.arm.genesis.build(n_envs=n_envs)

    def execute(self):
        self.arm.genesis.side_camera.start_recording()

        match self.pre_grasp_mode:
            case PreGraspMode.VERTICAL:
                self.move_to_ball_vertical()
            case PreGraspMode.VERTICAL_FLIP:
                self.move_to_ball_vertical_flip()
            case PreGraspMode.HORIZONTAL:
                target_z = 2 * Configuration.GOLF_BALL_RADIUS # TODO Path planning will push the ball away if not going through this pre-grasp position
                self.move_to_ball_horizontal(target_z)
                target_z = Configuration.GOLF_BALL_RADIUS
                self.move_to_ball_horizontal(target_z)
            case _:
                raise ValueError(f"Invalid pre-grasp mode: {self.pre_grasp_mode}")

        self.pick_qpos = self.arm.genesis.entity.get_dofs_position()
        self.LOGGER.info(f"pick frame joint configuration={self.pick_qpos}")
        self.move_to_cup()
        self.place_qpos = self.arm.genesis.entity.get_dofs_position()
        self.LOGGER.info(f"place frame joint configuration={self.place_qpos}")
        self.go_home()
        self.close_gripper()

        self.arm.genesis.side_camera.stop_recording("./so_arm_100.mp4", fps=self.arm.genesis.fps)
        return self.validate_success()

    def move_to_ball_vertical(self):
        target_pos, target_quat = self.vertical_3dpose(self.golf_ball)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_ball_vertical_flip(self):
        target_pos, target_quat = self.vertical_flip_3dpose(self.golf_ball)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_ball_horizontal(self, target_z):
        target_pos, target_quat = self.radial_3dpose(self.golf_ball, target_z)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_cup(self):
        target_z = SimPolicy.CUP_Z * Configuration.INCHES_TO_METERS
        target_pos, target_quat = self.radial_3dpose(self.cup, target_z)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_GRASPING_QPOS)

    def go_home(self):
        qpos = self.arm.genesis.home_qpos
        self.plan_path(qpos, SimPolicy.GRIPPER_OPENED_QPOS)

    def close_gripper(self):
        self.control_gripper(SimPolicy.GRIPPER_CLOSED_QPOS)

    def control_gripper(self, gripper_qpos):
        qpos = self.arm.genesis.entity.get_dofs_position()
        self.set_gripper_qpos(qpos, gripper_qpos)
        self.arm.genesis.entity.control_dofs_position(qpos)
        for _ in range(self.arm.genesis.fps * 1):
            self.arm.genesis.step()
            self.arm.genesis.side_camera.render()

    def ik_path_plan(self, target_pos, target_quat, gripper_qpos):
        self.LOGGER.info(f"IK target_pos={target_pos}")
        self.LOGGER.info(f"IK target_quat={target_quat}")

        target_pos = target_pos.unsqueeze(0)
        target_quat = target_quat.unsqueeze(0)

        qpos, ik_err = self.arm.genesis.entity.inverse_kinematics(
            link=self.arm.genesis.link, 
            pos=target_pos, 
            quat=target_quat,
            return_error=True
        )
        self.LOGGER.info(f"IK position error (m): {ik_err[0, :3]}")
        self.LOGGER.info(f"IK rotation error (rad): {ik_err[0, 3:]}")
        self.LOGGER.info(f"IK rotation error (deg): {torch.rad2deg(ik_err[0, 3:])}")
        #self.entity.control_dofs_position(qpos)

        self.plan_path(qpos, gripper_qpos)

        # Set tolerances
        pos_eps = 0.02  # 2cm position tolerance
        rot_eps = 0.1   # ~5.7 degrees rotation tolerance

        # Measure orientation error using angular distance
        link_quat = self.arm.genesis.link.get_quat()
        
        # Normalize quaternions
        link_quat = link_quat / torch.norm(link_quat, dim=-1, keepdim=True)
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
        
        # Compute angular distance (handles quaternion double cover: q and -q are same rotation)
        dot = torch.sum(link_quat * target_quat, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angular_error_rad = 2 * torch.arccos(torch.abs(dot))
        angular_error_deg = torch.rad2deg(angular_error_rad)
        
        self.LOGGER.info(f"Orientation angular error: {angular_error_rad.item():.4f} rad ({angular_error_deg.item():.2f}°)")
        if angular_error_rad > rot_eps:
            self.LOGGER.error(f"Orientation error is too large (tolerance: {rot_eps:.4f} rad = {torch.rad2deg(torch.tensor(rot_eps)):.2f}°)")

        # Measure position error
        link_pos = self.arm.genesis.link.get_pos()
        pos_diff = link_pos - target_pos
        self.LOGGER.info(f"Position difference: {pos_diff}")
        pos_error = torch.norm(pos_diff)
        self.LOGGER.info(f"Position error: {pos_error.item():.4f} m")
        if pos_error > pos_eps:
            self.LOGGER.error(f"Position error is too large (tolerance: {pos_eps:.4f} m)")

    def plan_path(self, qpos, gripper_qpos):
        path = self.arm.genesis.entity.plan_path(
            qpos,
        )

        for waypoint_qpos in path:
            self.set_gripper_qpos(waypoint_qpos, gripper_qpos)
            self.arm.genesis.entity.control_dofs_position(waypoint_qpos)
            self.arm.genesis.scene.step()
            self.arm.genesis.scene.clear_debug_objects()
            self.arm.draw_single_link_frame()
            self.arm.draw_link_arrow()
            self.arm.genesis.side_camera.render()

    def set_gripper_qpos(self, qpos, gripper_qpos):
        gripper_id = self.arm.genesis.joint.qs_idx_local[0]
        qpos[0][gripper_id] = gripper_qpos

    def tcp_pos(self):
        link_quat = self.arm.genesis.link.get_quat()
        tcp_offset_world = gu.transform_by_quat(self.arm.tcp_offset, link_quat)

        link_pos = self.arm.genesis.link.get_pos()
        return link_pos + tcp_offset_world

    def vertical_flip_3dpose(self, target_object):
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        quat_x = gu.axis_angle_to_quat(torch.tensor(torch.pi / 2), x_axis)  # -90° around x

        link_offset_world = gu.transform_by_quat(-self.arm.tcp_offset, quat_x)

        target_pos = target_object.get_pos()[0]
        target_link_pos = target_pos + link_offset_world

        return target_link_pos, quat_x

    def vertical_3dpose(self, target_object):
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        quat_x = gu.axis_angle_to_quat(torch.tensor(torch.pi / 2), x_axis)  # -90° around x

        y_axis = torch.tensor([0.0, 1.0, 0.0])
        quat_y = gu.axis_angle_to_quat(torch.tensor(torch.pi), y_axis)  # 180° around y

        down_quat = gu.transform_quat_by_quat(quat_y, quat_x)

        link_offset_world = gu.transform_by_quat(-self.arm.tcp_offset, down_quat)

        target_pos = target_object.get_pos()[0]
        target_link_pos = target_pos + link_offset_world

        return target_link_pos, down_quat

    def radial_3dpose(self, target_object, target_z):
        """Compute link pose: flip link first (180° around Y), then rotate around Z toward target."""
        rotation_pitch = self.arm.genesis.entity.get_link('Rotation_Pitch')

        rotation_pitch_pos = rotation_pitch.get_pos()[0]
        target_pos = target_object.get_pos()[0]

        target_pos[2] = target_z

        # Direction from rotation_pitch to target in XY plane
        radial_dir = target_pos - rotation_pitch_pos
        radial_dir_xy = radial_dir[:2]

        # 1. Flip link first (180° around Y)
        y_axis = torch.tensor([0.0, 1.0, 0.0])
        flip_quat = gu.axis_angle_to_quat(torch.tensor(torch.pi), y_axis)

        # 2. Then rotate around Z to align with radial direction
        z_axis = torch.tensor([0.0, 0.0, 1.0])
        radial_theta = torch.atan2(radial_dir_xy[0], -radial_dir_xy[1])

        rotate_quat = gu.axis_angle_to_quat(radial_theta, z_axis)
        rotate_quat = gu.transform_quat_by_quat(flip_quat, rotate_quat)

        link_offset_world = gu.transform_by_quat(-self.arm.tcp_offset, rotate_quat)
        target_link_pos = target_pos + link_offset_world

        radial_dir = target_link_pos - rotation_pitch_pos
        radial_dir_xy = radial_dir[:2]
        radial_theta = torch.atan2(radial_dir_xy[0], -radial_dir_xy[1])

        rotate_quat = gu.axis_angle_to_quat(radial_theta, z_axis)
        rotate_quat = gu.transform_quat_by_quat(flip_quat, rotate_quat)

        return target_link_pos, rotate_quat

    def validate_success(self) -> bool:
        """
        Check if the golf ball is near the cup position.
        
        Returns:
            True if ball is within distance threshold of cup, False otherwise
        """
        ball_pos = self.golf_ball.get_pos()[0]
        cup_pos = self.cup.get_pos()[0]
        
        self.LOGGER.info(f"Final ball position: {ball_pos}")
        self.LOGGER.info(f"Cup position: {cup_pos}")
        
        # Check distance in XY plane
        diff = ball_pos[:2] - cup_pos[:2]
        distance = torch.norm(diff)
        
        self.LOGGER.info(f"Distance from cup: {distance:.4f} meters")
        
        return distance < EpisodeReplayer.DISTANCE_THRESHOLD
