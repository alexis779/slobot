"""Merge FK gripper-link pose into observation.state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE

from slobot.hilserl.ee_kinematics import EE_STATE_DIM
from slobot.hilserl.factory import Factory
from slobot.hilserl.models.ee_state import EeObservation
from slobot.hilserl.models.motor_io import MotorSteps


@ProcessorStepRegistry.register("ee_pose_in_observation_state")
@dataclass
class EEPoseInObservationStateStep(ObservationProcessorStep):
    """Replace observation.state with normalized gripper_link pose + jaw motor rad."""

    def observation(self, observation: dict) -> dict:
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError(f"{OBS_STATE} is required before merging EE pose")

        motor_steps = MotorSteps.from_list(Factory.get_follower_feetech().get_pos())
        motor_rad = Factory.get_motors_to_radians().convert(motor_steps)

        bundle = get_kinematics_bundle_from_obs()
        pose, jaw_rad = bundle.fk.fk_to_gripper_pose(motor_rad)
        ee_obs = EeObservation(pose=pose, jaw_rad=jaw_rad)
        ee_tensor = ee_obs.to_tensor()

        new_observation = dict(observation)
        ee = ee_tensor.to(dtype=state.dtype, device=state.device)
        new_observation[OBS_STATE] = ee.reshape(*state.shape[:-1], EE_STATE_DIM)
        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(EE_STATE_DIM,),
        )
        return features


_cfg_holder: list = []
_motor_names_holder: list[str] = []


def set_kinematics_cfg(cfg, *, motor_names: list[str]) -> None:
    _cfg_holder.clear()
    _cfg_holder.append(cfg)
    _motor_names_holder.clear()
    _motor_names_holder.extend(motor_names)


def get_kinematics_bundle_from_obs():
    if not _cfg_holder or not _motor_names_holder:
        raise RuntimeError("Kinematics cfg not set; call set_kinematics_cfg from make_hilserl_processors")
    env_cfg = _cfg_holder[0]
    ik_cfg = env_cfg.processor.inverse_kinematics
    return Factory.get_kinematics_bundle(
        gripper_link_name=ik_cfg.gripper_link_name,
        jaw_joint_name=ik_cfg.jaw_joint_name,
        motor_names=_motor_names_holder,
        fps=env_cfg.fps,
    )
