import numpy as np
import torch
import time

from slobot.genesis import Genesis
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame
from slobot.feetech_frame import FeetechFrame
from slobot.feetech import Feetech

class SoArm100():
    LOGGER = Configuration.logger(__name__)

    # Mujoco home position
    HOME_QPOS = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]

    def sim_qpos(target_qpos):
        arm = SoArm100()
        arm.genesis.entity.set_dofs_position(target_qpos)
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

    def elemental_rotations(self):
        self.go_home()
        pos = self.genesis.fixed_jaw.get_pos()
        quat = self.genesis.fixed_jaw.get_quat()

        print("pos=", pos)
        print("quat=", quat)

        euler = self.genesis.quat_to_euler(quat)

        print("euler=", euler)

        steps = 2

        # turn the fixed jaw around the global x axis
        for roll in np.linspace(np.pi/2, 0, steps):
            euler[0] = roll
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat)

        # turn the fixed jaw around the global y axis
        for pitch in np.linspace(0, np.pi, steps):
            euler[1] = pitch
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat)

        # turn the fixed jaw around the global z axis
        pos = None
        for yaw in np.linspace(0, np.pi/2, steps):
            euler[2] = yaw
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.genesis.fixed_jaw, pos, quat)

    def go_home(self):
        target_qpos = torch.tensor(SoArm100.HOME_QPOS)
        self.genesis.follow_path(target_qpos)

    def handle_step(self) -> SimulationFrame:
        if self.step_handler is None:
            return

        simulation_frame = self.create_simulation_frame()
        self.step_handler.handle_step(simulation_frame)
        return simulation_frame

    def create_simulation_frame(self) -> SimulationFrame:
        current_time = time.time()

        qpos = self.genesis.entity.get_qpos()
        velocity = self.genesis.entity.get_dofs_velocity()
        force = self.genesis.entity.get_dofs_force()
        control_force = self.genesis.entity.get_dofs_control_force()

        simulation_frame = SimulationFrame(
            timestamp=current_time,
            control_pos=None,
            qpos=qpos,
            velocity=velocity,
            force=force,
            control_force=control_force,
        )

        if self.rgb or self.depth or self.segmentation or self.normal:
            frame = self.genesis.camera.render(rgb=self.rgb, depth=self.depth, segmentation=self.segmentation, colorize_seg=True, normal=self.normal)
            rbg_arr, depth_arr, seg_arr, normal_arr = frame
            simulation_frame.rgb = rbg_arr
            simulation_frame.depth = depth_arr
            simulation_frame.segmentation = seg_arr
            simulation_frame.normal = normal_arr

        if self.feetech is not None:
            simulation_frame.feetech_frame = self.feetech.create_feetech_frame()
        elif self.feetech_frame is not None:
            simulation_frame.feetech_frame = self.feetech_frame

        return simulation_frame

    def handle_qpos(self, feetech_frame: FeetechFrame):
        self.feetech_frame = feetech_frame
        self.genesis.entity.control_dofs_position(feetech_frame.control_pos)
        self.genesis.step()