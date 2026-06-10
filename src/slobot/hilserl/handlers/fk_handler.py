"""Forward kinematics via Genesis sim (gripper_link pose)."""

from __future__ import annotations

from slobot.hilserl.ee_kinematics import quat_wxyz_to_rotation_6d
from slobot.hilserl.handlers.motor_qpos import motor_to_sim_qpos
from slobot.hilserl.models.gripper_pose import GripperLinkPose
from slobot.hilserl.models.motor_io import MotorRadians
from slobot.robotic_arm import RoboticArm


class FkHandler:
    def __init__(
        self,
        robotic_arm: RoboticArm,
        *,
        n_dofs: int,
        jaw_joint_idx: int,
        gripper_link_name: str,
        jaw_joint_name: str,
    ):
        self._robotic_arm = robotic_arm
        self._n_dofs = n_dofs
        self._jaw_joint_idx = jaw_joint_idx
        self._entity = robotic_arm.genesis.entity
        self._gripper_link = self._entity.get_link(gripper_link_name)
        self.jaw_joint = self._entity.get_joint(jaw_joint_name)

    def fk_to_gripper_pose(self, radians: MotorRadians) -> tuple[GripperLinkPose, float]:
        """Set sim qpos from motor radians and return gripper_link pose + jaw rad."""
        qpos = motor_to_sim_qpos(radians, n_dofs=self._n_dofs)
        if self._entity.scene.n_envs > 0:
            qpos = qpos.unsqueeze(0)
        self._entity.set_dofs_position(qpos, zero_velocity=False)

        link_pos = self._gripper_link.get_pos()
        link_quat = self._gripper_link.get_quat()
        if link_pos.dim() > 1:
            link_pos = link_pos[0]
            link_quat = link_quat[0]

        pos = tuple(float(v) for v in link_pos.cpu().tolist())
        quat_wxyz = link_quat.cpu().tolist()
        rotation_6d = quat_wxyz_to_rotation_6d(quat_wxyz)

        current_qpos = self._entity.get_qpos()
        if current_qpos.dim() > 1:
            current_qpos = current_qpos[0]
        jaw_rad = float(current_qpos[self._jaw_joint_idx].item())

        return (
            GripperLinkPose(position=pos, rotation_6d=rotation_6d),
            jaw_rad,
        )
