"""Convert leader joint-position teleop actions for the HIL-SERL action pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep
from lerobot.types import EnvTransition, TransitionKey

from slobot.hilserl.models.motor_io import MotorSteps


@dataclass
class LeaderJointTeleopProcessorStep(ProcessorStep):
    """Map teleop joint dicts to an ordered float list for intervention recording."""

    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY)
        if isinstance(teleop_action, MotorSteps):
            complementary_data[TELEOP_ACTION_KEY] = teleop_action.tensor
        elif isinstance(teleop_action, dict) and teleop_action:
            if all(key.endswith(".pos") for key in teleop_action):
                complementary_data[TELEOP_ACTION_KEY] = [
                    teleop_action[f"{name}.pos"] for name in self.motor_names
                ]
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(self, features):
        return features
