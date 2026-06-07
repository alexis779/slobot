"""Processor steps for EE pose + gripper command actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import EnvTransition, PolicyAction, TransitionKey

from slobot.hilserl.ee_kinematics import (
    EE_STATE_DIM,
    EE_STATE_NAMES,
    clip_ee_xyz,
    limit_ee_position_step,
    normalize_ee_state,
)
from slobot.hilserl.handlers.gripper_command_handler import resolve_gripper_command
from slobot.hilserl.ee_pose_state_processor import get_kinematics_bundle_from_obs
from slobot.hilserl.factory import Factory
from slobot.hilserl.kinematics_bundle import KinematicsBundle
from slobot.hilserl.models.ee_state import EeAction
from slobot.hilserl.models.gripper_command import GripperCommand
from slobot.hilserl.models.gripper_pose import GripperLinkPose
from slobot.hilserl.models.motor_io import MotorRadians, MotorSteps

logger = logging.getLogger(__name__)


@dataclass
class PolicyEeActionProcessorStep(ProcessorStep):
    """Denormalize policy tensor to EeAction; clip XYZ and step limit."""

    motor_names: list[str]
    end_effector_bounds: dict | None = None
    max_ee_step_m: float = 0.05
    jaw_limits: tuple[float, float] = (0.0, 0.0)
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_intervention: bool = field(default=False, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
        if is_intervention:
            if not self._prev_intervention:
                self._sync_last_pos_from_actual()
            self._prev_intervention = True
            return transition
        if self._prev_intervention:
            self._sync_last_pos_from_actual()
        self._prev_intervention = False

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary.get("skip_hardware_action"):
            self._sync_last_pos_from_actual()

        ee_action = self._tensor_to_ee_action(action)
        if ee_action is None:
            return transition

        ee_action = self._apply_bounds(ee_action)
        new_transition = transition.copy()
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary["ee_action"] = ee_action
        complementary[TELEOP_ACTION_KEY] = ee_action.to_tensor()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def reset(self) -> None:
        self._last_pos = None
        self._prev_intervention = False

    def _sync_last_pos_from_actual(self) -> None:
        """Re-anchor step limiting to the real EE pose after a blocked command."""
        try:
            bundle = get_kinematics_bundle_from_obs()
        except RuntimeError:
            return
        feetech = Factory.get_follower_feetech()
        motor_rad = Factory.get_motors_to_radians().convert(
            MotorSteps.from_list(feetech.get_pos())
        )
        pose, _jaw_rad = bundle.fk.fk_to_gripper_pose(motor_rad)
        self._last_pos = np.array(pose.position, dtype=float)

    def _tensor_to_ee_action(self, action) -> EeAction | None:
        if isinstance(action, torch.Tensor):
            values = action.reshape(-1)
        elif isinstance(action, (list, tuple)):
            values = torch.tensor(action, dtype=torch.float32)
        else:
            return None
        if values.numel() != EE_STATE_DIM:
            return None
        return EeAction.from_tensor(values)

    def _apply_bounds(self, ee_action: EeAction) -> EeAction:
        x, y, z = ee_action.pose.position
        if self.end_effector_bounds is not None:
            x, y, z = clip_ee_xyz(x, y, z, self.end_effector_bounds)
        pos = np.array([x, y, z], dtype=float)
        pos = limit_ee_position_step(pos, self._last_pos, self.max_ee_step_m)
        self._last_pos = pos.copy()
        return EeAction(
            pose=GripperLinkPose(
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                rotvec=ee_action.pose.rotvec,
            ),
            command=ee_action.command,
        )

    def transform_features(self, features):
        return features


@dataclass
class SimCollisionGateProcessorStep(ProcessorStep):
    """IK + collision check for policy actions; penalize and skip hardware on collision."""

    bundle: KinematicsBundle
    motor_names: list[str]
    collision_penalty: float = -1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        if info.get(TeleopEvents.IS_INTERVENTION, False):
            new_transition = transition.copy()
            complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
            complementary.pop("skip_hardware_action", None)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
            return new_transition

        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        ee_action = complementary.get("ee_action")
        if not isinstance(ee_action, EeAction):
            return transition

        feetech = Factory.get_follower_feetech()
        q_guess = self._current_radians(transition, feetech)
        jaw_rad = resolve_gripper_command(
            ee_action.command,
            self.bundle.jaw_limits,
            feetech=feetech,
            jaw_motor_idx=self.bundle.jaw_motor_idx,
            q_guess=q_guess,
        )
        target_rad = self.bundle.ik.ik_from_gripper_pose(ee_action.pose, q_guess, jaw_rad)
        collision = self.bundle.collision.check_collision(target_rad)

        new_transition = transition.copy()
        new_info = dict(new_transition.get(TransitionKey.INFO, {}))

        if collision.valid:
            new_complementary = dict(complementary)
            new_complementary["target_motor_radians"] = target_rad
            new_complementary.pop("skip_hardware_action", None)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary
        else:
            logger.warning(
                "Collision detected (%d contact(s)); skipping hardware action",
                collision.contact_count,
            )
            new_transition[TransitionKey.REWARD] = float(
                new_transition.get(TransitionKey.REWARD, 0.0)
            ) + self.collision_penalty
            new_info["collision_invalid"] = True
            new_complementary = dict(complementary)
            new_complementary["skip_hardware_action"] = True
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary

        new_transition[TransitionKey.INFO] = new_info
        return new_transition

    def _current_radians(self, transition: EnvTransition, feetech) -> MotorRadians:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if isinstance(observation, dict):
            positions = [
                float(observation[f"{name}.pos"])
                for name in self.motor_names
                if f"{name}.pos" in observation
            ]
            if len(positions) == len(self.motor_names):
                return Factory.get_motors_to_radians().convert(MotorSteps.from_list(positions))
        return MotorRadians.from_list(feetech.get_qpos())

    def transform_features(self, features):
        return features


@dataclass
class EeActionToJointProcessorStep(ProcessorStep):
    """Convert policy IK targets to motor steps; passthrough teleop motor steps unchanged."""

    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)

        if complementary.get("skip_hardware_action") and not is_intervention:
            joint_positions = self._motor_steps_from_observation(transition)
            new_transition = transition.copy()
            dtype = action.dtype if isinstance(action, torch.Tensor) else torch.float32
            device = action.device if isinstance(action, torch.Tensor) else None
            new_transition[TransitionKey.ACTION] = torch.tensor(
                joint_positions, dtype=dtype, device=device
            )
            return new_transition

        target_rad = complementary.get("target_motor_radians")
        if target_rad is None:
            if is_intervention:
                values = self._action_values(action)
                if values is not None and len(values) == len(self.motor_names):
                    leader_rad = Factory.get_leader_feetech().pos_to_qpos(values)
                    joint_positions = self._qpos_to_motor_steps(leader_rad)
                    new_transition = transition.copy()
                    new_transition[TransitionKey.ACTION] = torch.tensor(
                        joint_positions, dtype=torch.float32
                    )
                    return new_transition
            return self._joints_from_list(action, transition)

        joint_positions = self._qpos_to_motor_steps(target_rad.to_list())
        dtype = action.dtype if isinstance(action, torch.Tensor) else torch.float32
        device = action.device if isinstance(action, torch.Tensor) else None
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.tensor(
            joint_positions, dtype=dtype, device=device
        )
        return new_transition

    def _joints_from_list(self, action, transition: EnvTransition) -> EnvTransition:
        values = self._action_values(action)
        if values is None or len(values) != len(self.motor_names):
            return transition
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.tensor(values, dtype=torch.float32)
        return new_transition

    def _qpos_to_motor_steps(self, qpos: list[float]) -> list[float]:
        return Factory.get_radians_to_motors().convert(MotorRadians.from_list(qpos)).to_list()

    def _motor_steps_from_observation(self, transition: EnvTransition) -> list[float]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        positions = [
            float(observation[f"{name}.pos"])
            for name in self.motor_names
            if f"{name}.pos" in observation
        ]
        if len(positions) == len(self.motor_names):
            return positions
        return Factory.get_follower_feetech().get_pos()

    def _action_values(self, action) -> list[float] | None:
        if isinstance(action, torch.Tensor):
            return [float(v) for v in action.reshape(-1).tolist()]
        if isinstance(action, (list, tuple)):
            return [float(v) for v in action]
        return None

    def transform_features(self, features):
        return features


@dataclass
class LeaderTeleopToEeActionProcessorStep(ProcessorStep):
    """FK leader arm joints to EE pose for dataset recording (teleop mode only)."""

    motor_names: list[str]
    bundle: KinematicsBundle

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        if not info.get(TeleopEvents.IS_INTERVENTION, False):
            return transition

        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary.pop("target_motor_radians", None)
        complementary.pop("skip_hardware_action", None)

        teleop_action = complementary.get(TELEOP_ACTION_KEY)
        joint_positions = self._joint_positions(teleop_action)
        if joint_positions is None:
            joint_positions = self._joint_positions(transition.get(TransitionKey.ACTION))
        if joint_positions is None:
            new_transition = transition.copy()
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
            return new_transition

        feetech = Factory.get_follower_feetech()
        leader_rad = MotorRadians.from_list(joint_positions)

        teleop_cmd = complementary.get("teleop_gripper_command", GripperCommand.STAY)
        if not isinstance(teleop_cmd, GripperCommand):
            teleop_cmd = GripperCommand.STAY

        previous_jaw_rad = None
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        gripper_key = f"{self.motor_names[self.bundle.jaw_motor_idx]}.pos"
        if gripper_key in observation:
            from slobot.hilserl.handlers.motor_qpos import jaw_rad_from_motor_step

            previous_jaw_rad = jaw_rad_from_motor_step(feetech, float(observation[gripper_key]))
        elif isinstance(observation, dict):
            raw = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("raw_joint_positions")
            if isinstance(raw, dict) and "gripper.pos" in raw:
                from slobot.hilserl.handlers.motor_qpos import jaw_rad_from_motor_step

                previous_jaw_rad = jaw_rad_from_motor_step(feetech, float(raw["gripper.pos"]))

        merged = leader_rad.to_list()
        merged[self.bundle.jaw_motor_idx] = resolve_gripper_command(
            teleop_cmd,
            self.bundle.jaw_limits,
            feetech=feetech,
            jaw_motor_idx=self.bundle.jaw_motor_idx,
            current_jaw_rad=previous_jaw_rad,
        )
        merged_rad = MotorRadians.from_list(merged)

        pose, _jaw_rad = self.bundle.fk.fk_to_gripper_pose(merged_rad)
        ee_action = EeAction(pose=pose, command=teleop_cmd)
        complementary["ee_action"] = ee_action
        complementary["target_motor_radians"] = merged_rad
        complementary.pop("skip_hardware_action", None)
        complementary[TELEOP_ACTION_KEY] = normalize_ee_state(
            [
                *pose.position,
                *pose.rotvec,
                teleop_cmd.to_normalized(),
            ]
        )
        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def _joint_positions(self, teleop_action) -> list[float] | None:
        leader_feetech = Factory.get_leader_feetech()
        if isinstance(teleop_action, MotorSteps):
            return leader_feetech.pos_to_qpos(teleop_action.to_list())
        if isinstance(teleop_action, torch.Tensor):
            return leader_feetech.pos_to_qpos(MotorSteps.from_tensor(teleop_action).to_list())
        if isinstance(teleop_action, (list, tuple)):
            return leader_feetech.pos_to_qpos(MotorSteps.from_list(teleop_action).to_list())
        if isinstance(teleop_action, dict) and teleop_action:
            if all(key.endswith(".pos") for key in teleop_action):
                pos = [float(teleop_action[f"{name}.pos"]) for name in self.motor_names]
                return leader_feetech.pos_to_qpos(pos)
        return None

    def transform_features(self, features):
        return features


@dataclass
class EeActionRecordProcessorStep(ProcessorStep):
    """Store normalized EeAction in complementary teleop_action for dataset recording."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        ee_action = complementary.get("ee_action")
        if isinstance(ee_action, EeAction):
            complementary[TELEOP_ACTION_KEY] = ee_action.to_tensor()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def transform_features(self, features):
        return features


EE_RECORD_ACTION_FEATURES = {
    "dtype": "float32",
    "shape": (len(EE_STATE_NAMES),),
    "names": list(EE_STATE_NAMES),
}
