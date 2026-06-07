"""Intervention step: leader joint teleop overrides policy actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import EnvTransition, PolicyAction, TransitionKey

from slobot.hilserl.models.motor_io import MotorSteps


@dataclass
class LeaderInterventionActionProcessorStep(ProcessorStep):
    """When teleop is active, replace policy action with leader motor steps."""

    motor_names: list[str]
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        if is_intervention and teleop_action is not None:
            action_list = self._teleop_to_action_list(teleop_action)
            if action_list is not None:
                teleop_action_tensor = torch.tensor(
                    action_list, dtype=action.dtype, device=action.device
                )
                new_transition[TransitionKey.ACTION] = teleop_action_tensor

        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def _teleop_to_action_list(self, teleop_action) -> list[float] | None:
        if isinstance(teleop_action, MotorSteps):
            return [float(v) for v in teleop_action.to_list()]
        if isinstance(teleop_action, dict) and teleop_action:
            if all(key.endswith(".pos") for key in teleop_action):
                return [float(teleop_action[f"{name}.pos"]) for name in self.motor_names]
            return None
        if isinstance(teleop_action, np.ndarray):
            return teleop_action.tolist()
        if isinstance(teleop_action, (list, tuple)):
            return [float(v) for v in teleop_action]
        if isinstance(teleop_action, torch.Tensor):
            return [float(v) for v in teleop_action.reshape(-1).tolist()]
        return None

    def transform_features(self, features):
        return features
