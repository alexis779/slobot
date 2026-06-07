"""Inverse kinematics via Genesis sim."""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from slobot.hilserl.handlers.motor_qpos import motor_to_sim_qpos, sim_to_motor_radians
from slobot.hilserl.models.gripper_pose import GripperLinkPose
from slobot.hilserl.models.motor_io import MotorRadians
from slobot.robotic_arm import RoboticArm


class IkHandler:
    def __init__(
        self,
        robotic_arm: RoboticArm,
        *,
        n_dofs: int,
        n_motors: int,
        jaw_joint_idx: int,
        gripper_link_name: str,
        jaw_joint_name: str,
    ):
        self._robotic_arm = robotic_arm
        self._n_dofs = n_dofs
        self._n_motors = n_motors
        self._jaw_joint_idx = jaw_joint_idx
        self._entity = robotic_arm.genesis.entity
        self._gripper_link = self._entity.get_link(gripper_link_name)
        self.jaw_joint = self._entity.get_joint(jaw_joint_name)

    def ik_from_gripper_pose(
        self,
        pose: GripperLinkPose,
        q_guess: MotorRadians,
        jaw_rad: float,
    ) -> MotorRadians:
        """IK for gripper_link pose; overwrite jaw DOF with jaw_rad."""
        target_pos = torch.tensor(pose.position, dtype=torch.float32)
        rotvec = torch.tensor(pose.rotvec, dtype=torch.float32)
        quat = ScipyRotation.from_rotvec(rotvec).as_quat(scalar_first=True)
        target_quat = torch.tensor(quat, dtype=torch.float32)

        if self._entity.scene.n_envs > 0:
            target_pos = target_pos.unsqueeze(0)
            target_quat = target_quat.unsqueeze(0)

        q_guess_tensor = motor_to_sim_qpos(q_guess, n_dofs=self._n_dofs)
        if self._entity.scene.n_envs > 0:
            q_guess_tensor = q_guess_tensor.unsqueeze(0)
        self._entity.set_dofs_position(q_guess_tensor, zero_velocity=False)

        target_qpos = self._entity.inverse_kinematics(
            link=self._gripper_link,
            pos=target_pos,
            quat=target_quat,
        )
        if target_qpos.dim() > 1:
            target_qpos = target_qpos[0]
        target_qpos = target_qpos.clone()
        target_qpos[self._jaw_joint_idx] = jaw_rad
        return sim_to_motor_radians(target_qpos, n_motors=self._n_motors, q_guess=q_guess)
