import numpy as np
import torch
import time

from slobot.genesis import Genesis
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame

class SoArm100():
    JAW_LENGTH = 0.107
    JAW_WIDTH = 0.008
    JAW_DEPTH = 0.0005

    # Mujoco home position
    HOME_QPOS = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]

    def sim_qpos(target_qpos):
        mjcf_path = Configuration.MJCF_CONFIG
        arm = SoArm100(mjcf_path=mjcf_path)
        arm.genesis.entity.set_qpos(target_qpos)
        arm.genesis.hold_entity()

    def __init__(self, **kwargs):
        self.step_handler = kwargs.get('step_handler', None)
        # overwrite step handler to delegate to this class first
        kwargs['step_handler'] = self

        self.genesis = Genesis(**kwargs)

        self.camera = self.genesis.camera
        self.entity = self.genesis.entity
        self.fixed_jaw = self.genesis.fixed_jaw

        self.rgb = kwargs.get('rgb', False)
        self.depth = kwargs.get('depth', False)
        self.segmentation = kwargs.get('segmentation', False)
        self.normal = kwargs.get('normal', False)

    def elemental_rotations(self):
        self.go_home()
        pos = self.fixed_jaw.get_pos()
        quat = self.fixed_jaw.get_quat()

        print("pos=", pos)
        print("quat=", quat)

        euler = self.genesis.quat_to_euler(quat)

        print("euler=", euler)

        steps = 2

        # turn the fixed jaw around the global x axis, from vertical to horizontal
        for roll in np.linspace(np.pi/2, 0, steps):
            euler[0] = roll
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.fixed_jaw, pos, quat)

        # turn the fixed jaw around the global y axis
        for pitch in np.linspace(0, np.pi, steps):
            euler[1] = pitch
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.fixed_jaw, pos, quat)

        # turn the fixed jaw around the global z axis
        pos = None
        for yaw in np.linspace(0, np.pi/2, steps):
            euler[2] = yaw
            quat = self.genesis.euler_to_quat(euler)
            self.genesis.move(self.fixed_jaw, pos, quat)

        self.stop()

    def stop(self):
        #self.camera.stop_recording(save_to_filename='so_arm_100.mp4')
        pass

    def go_home(self):
        target_qpos = torch.tensor(SoArm100.HOME_QPOS)
        self.genesis.follow_path(target_qpos)

    def draw_fixed_jaw_arrow(self):
        t = [self.JAW_WIDTH, -self.JAW_LENGTH, -self.JAW_DEPTH]
        self.genesis.draw_arrow(self.fixed_jaw, t)

    def open_jaw(self):
        self.genesis.update_qpos(self.jaw, np.pi/2)

    def handle_step(self):
        if self.step_handler is None:
            return

        current_time = time.time()

        # convert torch tensor to a JSON serializable object
        qpos = self.entity.get_qpos().tolist()

        simulation_frame = SimulationFrame(
            timestamp=current_time,
            qpos=qpos,
        )

        if self.rgb or self.depth or self.segmentation or self.normal:
            frame = self.camera.render(rgb=self.rgb, depth=self.depth, segmentation=self.segmentation, colorize_seg=True, normal=self.normal)
            rbg_arr, depth_arr, seg_arr, normal_arr = frame
            simulation_frame.rgb = rbg_arr
            simulation_frame.depth = depth_arr
            simulation_frame.segmentation = seg_arr
            simulation_frame.normal = normal_arr

        self.step_handler.handle_step(simulation_frame)

    def handle_qpos(self, qpos):
        self.entity.set_qpos(qpos)