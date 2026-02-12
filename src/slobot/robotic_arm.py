import numpy as np
import torch
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

        self.home_qpos = kwargs.get('home_qpos', None)

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
        for roll in np.linspace(np.pi/2, 0, steps):
            euler[0] = roll
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

        # turn the fixed jaw around the global y axis
        for pitch in np.linspace(0, np.pi, steps):
            euler[1] = pitch
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

        # turn the fixed jaw around the global z axis
        pos = None
        for yaw in np.linspace(0, np.pi/2, steps):
            euler[2] = yaw
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat.unsqueeze(0))

    def go_home(self):
        target_qpos = torch.tensor([self.home_qpos])
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