import torch
import genesis.utils.geom as gu

from slobot.robotic_arm import RoboticArm


class CartesianSimController:
    FRAME_SIZE = 0.1
    DPOS = 0.005
    DROT = 0.01

    def __init__(self, robotic_arm: RoboticArm):
        self.robotic_arm = robotic_arm
        self.genesis = robotic_arm.genesis
        self.entity = robotic_arm.genesis.entity

        link_names = [link.name for link in self.entity.links]
        self.target_link_name = link_names[-1]
        self._target_link = self.entity.get_link(self.target_link_name)
        self._sync_targets_from_link()

    def link_names(self) -> list[str]:
        return [link.name for link in self.entity.links]

    def joint_limits(self, joint_dof_idx: int) -> tuple[float, float]:
        min_limit, max_limit = self.entity.get_dofs_limit()
        return (
            min_limit[joint_dof_idx].item(),
            max_limit[joint_dof_idx].item(),
        )

    def link_pose(self, link_name: str | None = None) -> tuple[list[float], list[float]]:
        link = self._resolve_link(link_name)
        pos = link.get_pos()[0].tolist()
        euler = self.genesis.quat_to_euler(link.get_quat()[0].unsqueeze(0))
        euler = euler[0].tolist()
        return pos, euler

    def set_target_link(self, link_name: str):
        self._target_link = self.entity.get_link(link_name)
        self.sync_targets_from_link()

    def sync_targets_from_link(self):
        """Refresh cartesian targets from the current link pose in sim."""
        self._sync_targets_from_link()
        self.draw_link_frame()

    def translate_local(self, axis: int, direction: int) -> tuple[list[float], list[float]]:
        local_delta = torch.zeros(3, dtype=torch.float32)
        local_delta[axis] = direction * self.DPOS

        link_quat = self._target_link.get_quat()[0]
        world_delta = gu.transform_by_quat(local_delta, link_quat)
        self.target_pos = self.target_pos + world_delta

        return self._apply_target_pose()

    def rotate_local(self, axis: int, direction: int) -> tuple[list[float], list[float]]:
        axis_local = torch.zeros(3, dtype=torch.float32)
        axis_local[axis] = 1.0
        angle = torch.tensor(direction * self.DROT, dtype=torch.float32)
        drot_quat = gu.axis_angle_to_quat(angle, axis_local)
        # Body frame: R_new = R_target @ R_delta (matches RGB link axes)
        self.target_quat = gu.transform_quat_by_quat(drot_quat, self.target_quat)

        return self._apply_target_pose()

    def _apply_target_pose(self) -> tuple[list[float], list[float]]:
        target_pos = self.target_pos.unsqueeze(0)
        target_quat = self.target_quat.unsqueeze(0)

        qpos = self.entity.inverse_kinematics(
            link=self._target_link,
            pos=target_pos,
            quat=target_quat,
        )

        self.entity.control_dofs_position(qpos[0])
        self.genesis.step()
        self.draw_link_frame()
        self._sync_targets_from_link()

        return self.link_pose()

    def draw_link_frame(self):
        self.genesis.scene.clear_debug_objects()

        link_pos = self._target_link.get_pos()[0]
        link_quat = self._target_link.get_quat()[0]
        transform = gu.trans_quat_to_T(link_pos, link_quat)

        self.genesis.scene.draw_debug_frame(
            T=transform,
            axis_length=self.FRAME_SIZE,
            origin_size=0.01,
            axis_radius=0.005,
        )

    def _resolve_link(self, link_name: str | None):
        if link_name is None:
            return self._target_link
        return self.entity.get_link(link_name)

    def _sync_targets_from_link(self):
        pos, euler = self.link_pose()
        self.target_pos = torch.tensor(pos, dtype=torch.float32)
        self.target_quat = self.genesis.euler_to_quat(torch.tensor(euler, dtype=torch.float32))
