import torch
from importlib.resources import files

import genesis as gs
import genesis.utils.geom as gu

from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

class SimPolicy:
    """Policy to pick up a golf ball and place it in a cup using IK with vertical wrist constraint."""
    
    LOGGER = Configuration.logger(__name__)    

    INCHES_TO_METERS = 0.0254
    CUP_Z = 6 # inches

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
            ball_x * self.INCHES_TO_METERS,
            ball_y * self.INCHES_TO_METERS,
            Configuration.GOLF_BALL_RADIUS
        ])
        
        self.cup_pos = torch.tensor([
            cup_x * self.INCHES_TO_METERS,
            cup_y * self.INCHES_TO_METERS,
            0.0
        ])
        
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

        self.move_to_pregrasp()
        self.move_to_cup()
        self.open_gripper()

        self.arm.genesis.side_camera.stop_recording("./so_arm_100.mp4", fps=self.arm.genesis.fps)
        return self.validate_success()

    def move_to_pregrasp(self):
        target_quat = self.down_quat()

        tcp_offset_world = -gu.transform_by_quat(self.arm.tcp_offset.unsqueeze(0), target_quat)

        ball_pos = self.golf_ball.get_pos()
        target_pos = ball_pos + tcp_offset_world

        gripper_opened_qpos = 1.0
        return self.ik_path_plan(target_pos, target_quat, gripper_opened_qpos, rot_mask=[False, True, False])

    def move_to_cup(self):
        target_quat = self.radial_quat()

        tcp_offset_world = -gu.transform_by_quat(self.arm.tcp_offset, target_quat)
        target_pos = self.cup.get_pos() + tcp_offset_world

        target_pos[0][2] = target_pos[0][2] + SimPolicy.CUP_Z * self.INCHES_TO_METERS

        gripper_closed_qpos = 0.1
        return self.ik_path_plan(target_pos, target_quat, gripper_closed_qpos, rot_mask=[False, False, True])

    def open_gripper(self):
        qpos = self.arm.genesis.entity.get_dofs_position()
        self.set_gripper_qpos(qpos, 1.0)
        self.arm.genesis.entity.control_dofs_position(qpos)
        for step in range(self.arm.genesis.fps * 1):
            self.arm.genesis.scene.step()
            self.arm.genesis.side_camera.render()

    def ik_path_plan(self, target_pos, target_quat, gripper_qpos, rot_mask=[True, True, True]):

        self.LOGGER.info(f"IK target_pos={target_pos}")
        self.LOGGER.info(f"IK target_quat={target_quat}")

        qpos, ik_err = self.arm.genesis.entity.inverse_kinematics(
            link=self.arm.genesis.link, 
            pos=target_pos, 
            quat=target_quat,
            rot_mask=rot_mask,
            return_error=True
        )
        self.LOGGER.info(f"IK position error (m): {ik_err[0, :3]}")
        self.LOGGER.info(f"IK rotation error (rad): {ik_err[0, 3:]}")
        self.LOGGER.info(f"IK rotation error (deg): {torch.rad2deg(ik_err[0, 3:])}")
        #self.entity.control_dofs_position(qpos)

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

    def set_gripper_qpos(self, qpos, gripper_qpos):
        gripper_id = self.arm.genesis.joint.qs_idx_local[0]
        qpos[0][gripper_id] = gripper_qpos # open gripper 

    def tcp_pos(self):
        link_quat = self.arm.genesis.link.get_quat()
        tcp_offset_world = gu.transform_by_quat(self.arm.tcp_offset, link_quat)

        link_pos = self.arm.genesis.link.get_pos()
        return link_pos + tcp_offset_world

    def down_quat(self):
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        quat_x = gu.axis_angle_to_quat(torch.tensor(torch.pi / 2), x_axis)  # -90° around x

        y_axis = torch.tensor([0.0, 1.0, 0.0])
        quat_y = gu.axis_angle_to_quat(torch.tensor(torch.pi), y_axis)  # 180° around y

        return gu.transform_quat_by_quat(quat_y, quat_x).unsqueeze(0)

    def radial_quat(self):
        """Compute quaternion for link staying horizontal and facing radially toward the cup."""
        rotation_pitch = self.arm.genesis.entity.get_link('Rotation_Pitch')
        rotation_pitch_pos = rotation_pitch.get_pos()[0]
        cup_pos = self.cup.get_pos()[0]

        # Direction from rotation_pitch to cup in XY plane
        radial_dir = cup_pos - rotation_pitch_pos
        radial_dir_xy = radial_dir[:2]
        
        # Compute yaw angle from -Y axis to radial direction
        # Use atan2(x, -y) to measure angle from -Y axis
        yaw_angle = torch.atan2(radial_dir_xy[0], -radial_dir_xy[1])
        
        # Create yaw rotation around Z
        z_axis = torch.tensor([0.0, 0.0, 1.0])
        radial_quat = gu.axis_angle_to_quat(yaw_angle, z_axis)

        y_axis = torch.tensor([0.0, 1.0, 0.0])
        quat_y = gu.axis_angle_to_quat(torch.tensor(torch.pi), y_axis)  # 180° around y

        return gu.transform_quat_by_quat(quat_y, radial_quat).unsqueeze(0)

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
