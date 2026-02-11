from slobot.robotic_arm import RoboticArm
from slobot.configuration import Configuration
from importlib.resources import files

import torch
import genesis.utils.geom as gu

class SoArm100(RoboticArm):
    SO_ARM_100_MJCF_CONFIG = str(files('slobot.config') / "trs_so_arm100" / "so_arm100.xml") # "../mujoco_menagerie/trs_so_arm100/so_arm100.xml"

    DOFS = 6

    # Home position
    HOME_QPOS = [0, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, 0]

    GRIPPER_LINK_NAME = 'Fixed_Jaw'

    def __init__(self, **kwargs):
        kwargs['mjcf_path'] = SoArm100.SO_ARM_100_MJCF_CONFIG

        kwargs['home_qpos'] = SoArm100.HOME_QPOS

        kwargs['link_name'] = SoArm100.GRIPPER_LINK_NAME
        kwargs['camera_offset'] = self.camera_offset()

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

    def camera_offset(self):
        # 3 vertices from the STL 3d Mesh, measured in Blender, located at the holes of the mounting plate
        H1 = torch.tensor([-0.015826, -0.002098, -0.083203])  # bottom left
        H2 = torch.tensor([0.011174, -0.002098, -0.083203])  # bottom right
        H3 = torch.tensor([-0.015826, 0.00848, -0.060518])  # top left

        # Compute basis vectors for camera frame
        u = H2 - H1
        u_n = u / torch.linalg.norm(u)  # normalize X-axis

        v = H3 - H1
        # Orthogonalize v with respect to u (Gram-Schmidt)
        v = v - torch.dot(v, u) * u
        v_n = v / torch.linalg.norm(v)  # normalize Y-axis

        w = torch.cross(u, v, dim=0)  # Z-axis (camera optical axis)
        w_n = w / torch.linalg.norm(w)  # normalize
        w_n = -w_n

        # Camera anchor position (center of mounting plate)
        camera_anchor = H1 + (u + v) / 2

        # Build rotation matrix: columns are the basis vectors [X, Y, Z]
        camera_rotation = torch.column_stack([u_n, v_n, w_n])
        return gu.trans_R_to_T(camera_anchor, camera_rotation)
