import numpy as np
import torch
import time

from slobot.genesis import Genesis
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame

class SoArm100():
    # Mujoco home position
    HOME_QPOS = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]

    def sim_qpos(target_qpos):
        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path)
        arm.genesis.entity.set_qpos(target_qpos)
        arm.genesis.entity.control_dofs_position(target_qpos)
        arm.genesis.hold_entity()

    def __init__(self, **kwargs):
        self.step_handler = kwargs.get('step_handler', None)
        # overwrite step handler to delegate to this class first
        kwargs['step_handler'] = self

        self.feetech = kwargs.get('feetech', None)

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

        # turn the fixed jaw around the global x axis, from vertical to horizontal
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

    def diff_ik(self):
        center = torch.tensor([0, -0.1, 0.3])
        r = 0.1
        for i in range(0, 1000):
            target_pos = center + torch.tensor([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

            target_qpos = self.genesis.entity.inverse_kinematics(
                link     = self.genesis.fixed_jaw,
                pos      = target_pos,
                quat     = None,
            )

            self.genesis.entity.control_dofs_position(target_qpos)
            self.genesis.step()

    def lift_fixed_jaw(self):
        qpos_target = Configuration.QPOS_MAP['rotated']
        self.genesis.entity.control_dofs_position(qpos_target)

        for i in range(0, 100):
            self.genesis.step()

        print("qpos rotated=", self.genesis.entity.get_qpos())

        current_pos = self.genesis.fixed_jaw.get_pos()
        current_quat = self.genesis.fixed_jaw.get_quat()
        print(f"ee rotated pos={current_pos} quat={current_quat}")
        current_pos[2] += 0.1

        self.genesis.move(self.genesis.fixed_jaw, current_pos, None)

        print("qpos lifted=", self.genesis.entity.get_qpos())
        current_pos = self.genesis.fixed_jaw.get_pos()
        current_quat = self.genesis.fixed_jaw.get_quat()
        print(f"ee lifted pos={current_pos} quat={current_quat}")

        self.genesis.entity.control_dofs_position(self.genesis.entity.get_qpos())
        self.genesis.hold_entity()

    def stop(self):
        self.genesis.stop()

    def go_home(self):
        target_qpos = torch.tensor(SoArm100.HOME_QPOS)
        self.genesis.follow_path(target_qpos)

    def open_jaw(self):
        self.genesis.update_qpos(self.jaw, np.pi/2)

    def handle_step(self) -> SimulationFrame:
        if self.step_handler is None:
            return

        simulation_frame = self.create_simulation_frame()
        self.step_handler.handle_step(simulation_frame)
        return simulation_frame

    def create_simulation_frame(self) -> SimulationFrame:
        current_time = time.time()

        # convert torch tensor to a JSON serializable object
        qpos = self.genesis.entity.get_qpos().tolist()
        velocity = self.genesis.entity.get_dofs_velocity().tolist()
        force = self.genesis.entity.get_dofs_force().tolist()
        control_force = self.genesis.entity.get_dofs_control_force().tolist()

        simulation_frame = SimulationFrame(
            timestamp=current_time,
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

        return simulation_frame

    def handle_qpos(self, feetech_frame):
        self.genesis.entity.control_dofs_position(feetech_frame.qpos)
        self.genesis.step()