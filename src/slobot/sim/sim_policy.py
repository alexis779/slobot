import torch

import genesis.utils.geom as gu

from slobot.configuration import Configuration
from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.sim.recording_layout import PreGraspMode, RecordingLayout
from slobot.sim.ompl_path_planner import OMPLPathPlanner

class SimPolicy:
    """Policy to pick up a golf ball and place it in a cup using IK with wrist orientation constraint."""
    
    LOGGER = Configuration.logger(__name__)    

    CUP_Z = 6 # inches

    GRIPPER_OPENED_QPOS = 1.0
    GRIPPER_GRASPING_QPOS = 0.25
    GRIPPER_CLOSED_QPOS = -1.0

    def __init__(self, golf_ball_env: GolfBallEnv):
        self.golf_ball_env = golf_ball_env
        self.ompl_path_planner = OMPLPathPlanner(golf_ball_env.arm.genesis.entity, golf_ball_env.arm.genesis.scene)

    def execute(self, recording_layout: RecordingLayout) -> bool:
        self.golf_ball_env.arm.genesis.side_camera.start_recording()
        try:
            return self.execute_policy(recording_layout)
        finally:
            self.golf_ball_env.arm.genesis.side_camera.stop_recording()

    def execute_policy(self, recording_layout: RecordingLayout) -> bool:
        self.golf_ball_env.set_object_initial_positions(recording_layout.ball_x, recording_layout.ball_y, recording_layout.cup_x, recording_layout.cup_y)

        match recording_layout.pre_grasp_mode:
            case PreGraspMode.VERTICAL:
                self.move_to_ball_vertical()
            case PreGraspMode.VERTICAL_FLIP:
                self.move_to_ball_vertical_flip()
            case PreGraspMode.HORIZONTAL:
                self.move_to_ball_horizontal()
            case _:
                raise ValueError(f"Invalid pre-grasp mode: {recording_layout.pre_grasp_mode}")

        self.pick_link_quat = self.golf_ball_env.arm.genesis.link.get_quat()
        self.grasp()
        self.pick_qpos = self.golf_ball_env.arm.genesis.entity.get_dofs_position()
        self.LOGGER.info(f"pick frame joint configuration={self.pick_qpos}")
        #input("Press Enter to continue...")
        #self.golf_ball_env.arm.genesis.hold_entity()
        self.move_to_cup()
        self.place_qpos = self.golf_ball_env.arm.genesis.entity.get_dofs_position()
        self.LOGGER.info(f"place frame joint configuration={self.place_qpos}")
        self.go_home()
        self.close_gripper()
        self.golf_ball_env.arm.genesis.scene.clear_debug_objects()

        success = self.golf_ball_env.is_golf_ball_in_cup()
        if not success:
            raise ValueError("Golf ball not in cup")

        return success

    def move_to_ball_vertical(self):
        target_pos, target_quat = self.vertical_3dpose(self.golf_ball_env.golf_ball)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_ball_vertical_flip(self):
        target_pos, target_quat = self.vertical_flip_3dpose(self.golf_ball_env.golf_ball)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_ball_horizontal(self):
        target_z = Configuration.GOLF_BALL_RADIUS
        target_pos, target_quat = self.radial_3dpose(self.golf_ball_env.golf_ball, target_z)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_OPENED_QPOS)

    def move_to_cup(self):
        target_z = SimPolicy.CUP_Z * Configuration.INCHES_TO_METERS
        target_pos, target_quat = self.radial_3dpose(self.golf_ball_env.cup, target_z)

        return self.ik_path_plan(target_pos, target_quat, SimPolicy.GRIPPER_GRASPING_QPOS)

    def go_home(self):
        qpos = self.golf_ball_env.arm.genesis.home_qpos
        self.plan_path(qpos, SimPolicy.GRIPPER_OPENED_QPOS)

    def grasp(self):
        self.control_gripper(SimPolicy.GRIPPER_GRASPING_QPOS)

    def close_gripper(self):
        self.control_gripper(SimPolicy.GRIPPER_CLOSED_QPOS)

    def control_gripper(self, gripper_qpos):
        qpos = self.golf_ball_env.arm.genesis.entity.get_dofs_position()
        self.set_gripper_qpos(qpos[0], gripper_qpos)
        self.golf_ball_env.arm.genesis.entity.control_dofs_position(qpos)
        self.stabilize()

    def stabilize(self):
        for _ in range(self.golf_ball_env.arm.genesis.fps * 1):
            #input("Press Enter to step...")
            self.golf_ball_env.arm.genesis.step()
            self.golf_ball_env.arm.genesis.side_camera.render()

    def ik_path_plan(self, target_pos, target_quat, gripper_qpos):
        self.LOGGER.info(f"IK target_pos={target_pos}")
        self.LOGGER.info(f"IK target_quat={target_quat}")

        target_pos = target_pos.unsqueeze(0)
        target_quat = target_quat.unsqueeze(0)

        qpos, ik_err = self.golf_ball_env.arm.genesis.entity.inverse_kinematics(
            link=self.golf_ball_env.arm.genesis.link, 
            pos=target_pos, 
            quat=target_quat,
            return_error=True
        )

        # Set tolerances
        pos_eps = 0.02  # position tolerance
        pos_error = torch.norm(ik_err[0, :3])
        self.LOGGER.info(f"IK position error (m): {pos_error}")
        if pos_error > pos_eps:
            self.LOGGER.warning(f"IK position error is too large {pos_error} > {pos_eps}")

        self.LOGGER.info(f"IK rotation error (rad): {ik_err[0, 3:]}")
        self.LOGGER.info(f"IK rotation error (deg): {torch.rad2deg(ik_err[0, 3:])}")
        #self.entity.control_dofs_position(qpos)

        self.plan_path(qpos, gripper_qpos)
        self.stabilize()

        rot_eps = 0.1   # ~5.7 degrees rotation tolerance

        # Measure orientation error using angular distance
        link_quat = self.golf_ball_env.arm.genesis.link.get_quat()
        
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
            raise ValueError(f"Orientation error is too large {angular_error_rad} > {rot_eps}")

        # Measure position error
        link_pos = self.golf_ball_env.arm.genesis.link.get_pos()
        pos_diff = link_pos - target_pos
        self.LOGGER.info(f"Position difference: {pos_diff}")
        pos_error = torch.norm(pos_diff)
        self.LOGGER.info(f"Position error: {pos_error.item():.4f} m")
        if pos_error > pos_eps:
            raise ValueError(f"Position error is too large {pos_error} > {pos_eps}")

    def plan_path(self, qpos, gripper_qpos):
        '''
        path, valid_mask = self.golf_ball_env.arm.genesis.entity.plan_path(
            qpos,
            return_valid_mask=True,
        )
        if not valid_mask.all():
            raise ValueError("Path planning failed")
        '''
        path = self.ompl_path_planner.plan(qpos)
        self.LOGGER.info(f"path start: {path[0]}")
        self.LOGGER.info(f"path end: {path[-1]}")
        self.LOGGER.info(f"path before interpolation has {path.shape[0]} waypoints")
        self.LOGGER.info(f"path = {path}")
        path = self.ompl_path_planner.interpolate(path, path_length=200)

        for waypoint_qpos in path:
            self.set_gripper_qpos(waypoint_qpos, gripper_qpos)
            self.golf_ball_env.arm.genesis.entity.control_dofs_position(waypoint_qpos)
            self.golf_ball_env.arm.genesis.scene.step()
            self.golf_ball_env.arm.genesis.scene.clear_debug_objects()
            self.golf_ball_env.arm.draw_single_link_frame()
            self.golf_ball_env.arm.draw_arrow_from_link_to_tcp()
            self.golf_ball_env.arm.genesis.side_camera.render()

    def set_gripper_qpos(self, qpos, gripper_qpos):
        gripper_id = self.golf_ball_env.arm.genesis.joint.qs_idx_local[0]
        qpos[gripper_id] = gripper_qpos

    def tcp_pos(self):
        link_quat = self.golf_ball_env.arm.genesis.link.get_quat()
        tcp_offset_world = gu.transform_by_quat(self.golf_ball_env.arm.tcp_offset, link_quat)

        link_pos = self.golf_ball_env.arm.genesis.link.get_pos()
        return link_pos + tcp_offset_world

    def vertical_flip_3dpose(self, target_object):
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        quat_x = gu.axis_angle_to_quat(torch.tensor(torch.pi / 2), x_axis)  # -90° around x

        link_offset_world = gu.transform_by_quat(-self.golf_ball_env.arm.tcp_offset, quat_x)

        target_pos = target_object.get_pos()[0]

        target_link_pos = target_pos + link_offset_world

        return target_link_pos, quat_x

    def vertical_3dpose(self, target_object, target_z):
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        quat_x = gu.axis_angle_to_quat(torch.tensor(torch.pi / 2), x_axis)  # -90° around x

        y_axis = torch.tensor([0.0, 1.0, 0.0])
        quat_y = gu.axis_angle_to_quat(torch.tensor(torch.pi), y_axis)  # 180° around y

        down_quat = gu.transform_quat_by_quat(quat_y, quat_x)

        link_offset_world = gu.transform_by_quat(-self.golf_ball_env.arm.tcp_offset, down_quat)

        target_pos = target_object.get_pos()[0]
        target_pos[2] = target_z

        target_link_pos = target_pos + link_offset_world

        return target_link_pos, down_quat

    def radial_3dpose(self, target_object, target_z):
        """Compute link pose: flip link first (180° around Y), then rotate around Z toward target."""
        rotation_pitch = self.golf_ball_env.arm.genesis.entity.get_link('Rotation_Pitch')

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

        link_offset_world = gu.transform_by_quat(-self.golf_ball_env.arm.tcp_offset, rotate_quat)
        target_link_pos = target_pos + link_offset_world

        radial_dir = target_link_pos - rotation_pitch_pos
        radial_dir_xy = radial_dir[:2]
        radial_theta = torch.atan2(radial_dir_xy[0], -radial_dir_xy[1])

        rotate_quat = gu.axis_angle_to_quat(radial_theta, z_axis)
        rotate_quat = gu.transform_quat_by_quat(flip_quat, rotate_quat)

        return target_link_pos, rotate_quat