import torch
import genesis.utils.geom as gu

import time
from functools import cached_property

from slobot.genesis import Genesis
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame, CameraFrame
from slobot.feetech_frame import FeetechFrame
from slobot.feetech import Feetech


class RoboticArm():
    LOGGER = Configuration.logger(__name__)

    def sim_qpos(arm, target_qpos):
        arm.genesis.entity.control_dofs_position(target_qpos)
        arm.genesis.hold_entity()

    def __init__(self, **kwargs):
        self.step_handler = kwargs.get('step_handler', None)
        # overwrite step handler to delegate to this class first
        kwargs['step_handler'] = self

        self.feetech : Feetech = kwargs.get('feetech', None)
        self.feetech_frame : FeetechFrame = None

        self.genesis = Genesis(**kwargs)

        self.rgb = kwargs.get('rgb', False)
        self.depth = kwargs.get('depth', False)
        self.segmentation = kwargs.get('segmentation', False)
        self.normal = kwargs.get('normal', False)

    @cached_property
    def joint_names(self):
        return [
            joint.name for joint in self.genesis.entity.joints
        ]

    def elemental_rotations(self):
        self.go_home()
        pos = self.genesis.fixed_jaw.get_pos()
        quat = self.genesis.fixed_jaw.get_quat()

        print("pos=", pos)
        print("quat=", quat)

        euler = self.genesis.quat_to_euler(quat)
        euler = euler[0]

        print("euler=", euler)

        steps = 2

        # turn the fixed jaw around the global x axis
        for roll in torch.linspace(torch.pi/2, 0, steps):
            euler[0] = roll
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

        # turn the fixed jaw around the global y axis
        for pitch in torch.linspace(0, torch.pi, steps):
            euler[1] = pitch
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

        # turn the fixed jaw around the global z axis
        pos = None
        for yaw in torch.linspace(0, torch.pi/2, steps):
            euler[2] = yaw
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

    def go_home(self):
        target_qpos = self.genesis.home_qpos
        self.genesis.follow_path(target_qpos)

    def handle_step(self) -> SimulationFrame:
        if self.step_handler is None:
            return

        simulation_frame = self.create_simulation_frame()
        self.step_handler.handle_step(simulation_frame)
        return simulation_frame

    def create_simulation_frame(self) -> SimulationFrame:
        current_time = time.time()

        qpos = self.genesis.entity.get_qpos()[0].tolist()
        velocity = self.genesis.entity.get_dofs_velocity()[0].tolist()
        force = self.genesis.entity.get_dofs_force()[0].tolist()
        control_force = self.genesis.entity.get_dofs_control_force()[0].tolist()

        simulation_frame = SimulationFrame(
            timestamp=current_time,
            control_pos=None,
            qpos=qpos,
            velocity=velocity,
            force=force,
            control_force=control_force,
            side_camera_frame=CameraFrame(),
            link_camera_frame=CameraFrame(),
        )

        if self.rgb or self.depth or self.segmentation or self.normal:
            frame = self.genesis.side_camera.render(rgb=self.rgb, depth=self.depth, segmentation=self.segmentation, colorize_seg=True, normal=self.normal)
            rbg_arr, depth_arr, seg_arr, normal_arr = frame
            simulation_frame.side_camera_frame.rgb = rbg_arr
            simulation_frame.side_camera_frame.depth = depth_arr
            simulation_frame.side_camera_frame.segmentation = seg_arr
            simulation_frame.side_camera_frame.normal = normal_arr

            if self.genesis.link_camera:
                frame = self.genesis.link_camera.render(rgb=self.rgb, depth=self.depth, segmentation=self.segmentation, colorize_seg=True, normal=self.normal)
                rbg_arr, depth_arr, seg_arr, normal_arr = frame
                simulation_frame.link_camera_frame.rgb = rbg_arr
                simulation_frame.link_camera_frame.depth = depth_arr
                simulation_frame.link_camera_frame.segmentation = seg_arr
                simulation_frame.link_camera_frame.normal = normal_arr

        if self.feetech is not None:
            simulation_frame.feetech_frame = self.feetech.create_feetech_frame()
        elif self.feetech_frame is not None:
            simulation_frame.feetech_frame = self.feetech_frame

        return simulation_frame

    def handle_qpos(self, feetech_frame: FeetechFrame):
        self.feetech_frame = feetech_frame
        self.genesis.entity.control_dofs_position(feetech_frame.control_pos)
        self.genesis.step()

    def draw_link_arrow(self):
        link_pos = self.genesis.link.get_pos()

        link_quat = self.genesis.link.get_quat()
        tcp_offset_world = gu.transform_by_quat(self.tcp_offset, link_quat)

        for env_idx in range(self.genesis.scene.n_envs):
            self.genesis.scene.draw_debug_arrow(torch.from_numpy(self.genesis.scene.envs_offset[env_idx]) + link_pos[env_idx], tcp_offset_world[env_idx], color=(1, 1, 1), radius=0.005)

    def draw_single_link_frame(self, frame_size=0.1):
        """Draw coordinate frame for a specific link.

        Args:
            frame_size: Length of the frame axes in meters
        """
        # Get link position and orientation for all environments
        link_pos = self.genesis.link.get_pos()
        link_quat = self.genesis.link.get_quat()

        # Create frame axes (X=red, Y=green, Z=blue)
        x_axis = torch.tensor([frame_size, 0, 0])
        y_axis = torch.tensor([0, frame_size, 0])
        z_axis = torch.tensor([0, 0, frame_size])

        for env_idx in range(self.genesis.scene.n_envs):
            # Transform axes by link orientation
            x_axis_world = gu.transform_by_quat(x_axis, link_quat[env_idx])
            y_axis_world = gu.transform_by_quat(y_axis, link_quat[env_idx])
            z_axis_world = gu.transform_by_quat(z_axis, link_quat[env_idx])

            # Get link position in world frame (with environment offset)
            origin = torch.from_numpy(self.genesis.scene.envs_offset[env_idx]) + link_pos[env_idx]

            # Draw the three axes
            self.genesis.scene.draw_debug_arrow(origin, x_axis_world, color=(1, 0, 0), radius=0.005)  # X-axis in red
            self.genesis.scene.draw_debug_arrow(origin, y_axis_world, color=(0, 1, 0), radius=0.005)  # Y-axis in green
            self.genesis.scene.draw_debug_arrow(origin, z_axis_world, color=(0, 0, 1), radius=0.005)  # Z-axis in blue
