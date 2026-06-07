"""Inject leader teleop gripper command into transition complementary data."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.processor.pipeline import ProcessorStep
from lerobot.types import EnvTransition, TransitionKey

from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop


@dataclass
class TeleopGripperCommandProcessorStep(ProcessorStep):
    teleop_device: SlobotSO100LeaderTeleop

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary["teleop_gripper_command"] = self.teleop_device.gripper_command
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def transform_features(self, features):
        return features
