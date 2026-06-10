"""Reward classifier step that applies the saved training preprocessor at inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGE

from lerobot.processor.hil_processor import RewardClassifierProcessorStep
from lerobot.processor.pipeline import DataProcessorPipeline, PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.processor.pipeline import EnvTransition, TransitionKey

REWARD_PROBABILITY = "next.reward.probability"

logger = logging.getLogger(__name__)


def find_reward_classifier_step(
    env_processor: DataProcessorPipeline,
) -> SlobotRewardClassifierProcessorStep | None:
    """Return the slobot reward-classifier step from an env processor pipeline."""
    for step in env_processor.steps:
        if isinstance(step, SlobotRewardClassifierProcessorStep):
            return step
    return None


@dataclass
@ProcessorStepRegistry.register("slobot_reward_classifier_processor")
class SlobotRewardClassifierProcessorStep(RewardClassifierProcessorStep):
    """Applies classifier_preprocessor.json before predict_reward (LeRobot omits this)."""

    preprocessor: PolicyProcessorPipeline | None = field(default=None, repr=False)

    def _should_terminate_on_classifier_success(self, transition: EnvTransition) -> bool:
        if not self.terminate_on_success:
            return False
        info = transition.get(TransitionKey.INFO, {})
        if info.get(TeleopEvents.IS_INTERVENTION, False):
            return False
        return True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pretrained_path is not None and self.reward_classifier is not None:
            self.preprocessor = PolicyProcessorPipeline.from_pretrained(
                self.pretrained_path,
                config_filename="classifier_preprocessor.json",
                overrides={"device_processor": {"device": self.device}},
            )

    def predict_probability_from_observation(
        self,
        observation: dict[str, Any],
    ) -> float | None:
        """Run the classifier on an observation dict (same images saved to the dataset)."""
        if self.reward_classifier is None or self.preprocessor is None:
            return None

        expected_keys = list(self.reward_classifier.config.input_features)
        batch: dict[str, torch.Tensor] = {}
        for key in expected_keys:
            if key not in observation:
                return None
            value = observation[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            batch[key] = value

        with torch.inference_mode():
            batch = self.preprocessor(batch)
            images = [
                batch[key]
                for key in self.reward_classifier.config.input_features
                if key.startswith(OBS_IMAGE)
            ]
            return float(self.reward_classifier.predict(images).probabilities.squeeze().cpu())

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None or self.preprocessor is None:
            return new_transition

        expected_keys = list(self.reward_classifier.config.input_features)
        if not all(key in observation for key in expected_keys):
            missing = [key for key in expected_keys if key not in observation]
            present = [key for key in observation if "image" in key]
            logger.warning(
                "Reward classifier skipped: missing observation keys %s (have image keys %s)",
                missing,
                present,
            )
            return new_transition

        start_time = time.perf_counter()
        success_prob = self.predict_probability_from_observation(observation)
        if success_prob is None:
            return new_transition
        success = success_prob >= self.success_threshold

        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)
        info = new_transition.get(TransitionKey.INFO, {})
        is_teleop = bool(info.get(TeleopEvents.IS_INTERVENTION, False))

        if success:
            reward = self.success_reward
            will_terminate = self._should_terminate_on_classifier_success(new_transition)
            if will_terminate:
                terminated = True
            logger.info(
                "Reward classifier success: prob=%.4f reward=%.1f teleop=%s terminate=%s",
                success_prob,
                self.success_reward,
                is_teleop,
                will_terminate,
            )
            if is_teleop and self.terminate_on_success and not will_terminate:
                logger.info(
                    "Reward classifier success during teleop: reward=%.1f applied, "
                    "episode continues (end with q or return to policy)",
                    self.success_reward,
                )

        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        info["reward_classifier_frequency"] = 1 / (time.perf_counter() - start_time)
        info[REWARD_PROBABILITY] = success_prob
        new_transition[TransitionKey.INFO] = info

        if not success and success_prob > 0.1:
            logger.info(
                "Reward classifier prob=%.4f (threshold=%.2f) teleop=%s",
                success_prob,
                self.success_threshold,
                is_teleop,
            )

        return new_transition
