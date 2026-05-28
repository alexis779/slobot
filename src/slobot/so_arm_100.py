from slobot.robotic_arm import RoboticArm
from slobot.configuration import Configuration

from functools import cached_property

import torch
import genesis.utils.geom as gu

class SoArm100(RoboticArm):
    DOFS = 6

    MODEL_RESOLUTION = 4096

    # the translation vector from the gripper link position to tool center point, in the frame relative to the link
    TCP_OFFSET = [-1.5e-2, -9e-2, 0.5e-2]

    def __init__(self, **kwargs):
        kwargs['mjcf_path'] = Configuration.MJCF_CONFIG
        kwargs['link_name'] = Configuration.MJCF_GRIPPER_LINK_NAME
        kwargs['joint_name'] = Configuration.MJCF_GRIPPER_JOINT_NAME
        self.qpos_map = Configuration.MJCF_QPOS_MAP

        #kwargs['urdf_path'] = Configuration.URDF_CONFIG
        #kwargs['link_name'] = Configuration.URDF_GRIPPER_LINK_NAME
        #kwargs['joint_name'] = Configuration.URDF_GRIPPER_JOINT_NAME
        #self.qpos_map = Configuration.URDF_QPOS_MAP

        kwargs['camera_offset'] = self.camera_offset()

        super().__init__(**kwargs)

    def preset_qpos(self, preset):
        return self.qpos_map[preset]

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

    @cached_property
    def tcp_offset(self):
        return torch.tensor(SoArm100.TCP_OFFSET)