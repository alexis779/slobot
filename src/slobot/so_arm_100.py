from slobot.robotic_arm import RoboticArm
from slobot.configuration import Configuration
from importlib.resources import files
import torch

class SoArm100(RoboticArm):
    SO_ARM_100_MJCF_CONFIG = str(files('slobot.config') / "trs_so_arm100" / "so_arm100.xml") # "../mujoco_menagerie/trs_so_arm100/so_arm100.xml"

    DOFS = 6

    # Home position
    HOME_QPOS = [0, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, 0]

    def __init__(self, **kwargs):
        kwargs['mjcf_path'] = SoArm100.SO_ARM_100_MJCF_CONFIG
        super().__init__(**kwargs)
        self.set_kinematic_path()

    def set_kinematic_path(self):
        '''Kinematic path'''
        self.base = self.genesis.entity.get_link('Base')
        self.shoulder_pan = self.genesis.entity.get_joint('Rotation')
        self.rotation_pitch = self.genesis.entity.get_link('Rotation_Pitch')
        self.shoulder_lift = self.genesis.entity.get_joint('Pitch')
        self.upper_arm = self.genesis.entity.get_link('Upper_Arm')
        self.elbow_flex = self.genesis.entity.get_joint('Elbow')
        self.lower_arm = self.genesis.entity.get_link('Lower_Arm')
        self.wrist_flex = self.genesis.entity.get_joint('Wrist_Pitch')
        self.wrist_pitch_roll = self.genesis.entity.get_link('Wrist_Pitch_Roll')
        self.wrist_roll = self.genesis.entity.get_joint('Wrist_Roll')
        self.fixed_jaw = self.genesis.entity.get_link('Fixed_Jaw')
        self.gripper = self.genesis.entity.get_joint('Jaw')
        self.moving_jaw = self.genesis.entity.get_link('Moving_Jaw')

    def home_qpos(self):
        return SoArm100.HOME_QPOS
