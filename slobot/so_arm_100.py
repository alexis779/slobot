import numpy as np

from slobot.genesis import Genesis
from slobot.configuration import Configuration

class SoArm100():
    JAW_LENGTH = 0.107
    JAW_WIDTH = 0.008
    JAW_DEPTH = 0.0005

    # Mujoco home position
    HOME_QPOS = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]

    def sim_qpos(target_qpos):
        mjcf_path = '../mujoco_menagerie/trs_so_arm100/so_arm100.xml'
        arm = SoArm100(mjcf_path=mjcf_path)
        arm.genesis.entity.set_qpos(target_qpos)
        arm.genesis.hold_entity()

    def __init__(self, **kwargs):
        self.qpos_handler = kwargs.get('qpos_handler', None)

        # provide a callback for each step update of the simulation
        kwargs['step_handler'] = self
        self.genesis = Genesis(**kwargs)

        self.camera = self.genesis.camera
        self.entity = self.genesis.entity
        self.fixed_jaw = self.genesis.fixed_jaw

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
        self.camera.stop_recording(save_to_filename='so_arm_100.mp4')

    def go_home(self):
        self.genesis.follow_path(SoArm100.HOME_QPOS)
    
    def draw_fixed_jaw_arrow(self):
        t = [self.JAW_WIDTH, -self.JAW_LENGTH, -self.JAW_DEPTH]
        self.genesis.draw_arrow(self.fixed_jaw, t)

    def open_jaw(self):
        self.genesis.update_qpos(self.jaw, np.pi/2)
    
    def step(self):
        self.camera.render()
        if self.qpos_handler is not None:
            self.qpos_handler.set_qpos(self.entity.get_qpos())

