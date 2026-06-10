"""Reward classifier step that applies the saved training preprocessor at inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGE

from lerobot.processor.hil_processor import RewardClassifierProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.processor.pipeline import EnvTransition, TransitionKey

REWARD_PROBABILITY = "next.reward.probability"

logger = logging.getLogger(__name__)


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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None or self.preprocessor is None:
            return new_transition

        expected_keys = list(self.reward_classifier.config.input_features)
        batch = {key: observation[key] for key in expected_keys if key in observation}
        if len(batch) != len(expected_keys):
            missing = [key for key in expected_keys if key not in batch]
            present = [key for key in observation if "image" in key]
            logger.warning(
                "Reward classifier skipped: missing observation keys %s (have image keys %s)",
                missing,
                present,
            )
            return new_transition

        start_time = time.perf_counter()
        with torch.inference_mode():
            batch = self.preprocessor(batch)
            images = [
                batch[key]
                for key in self.reward_classifier.config.input_features
                if key.startswith(OBS_IMAGE)
            ]
            success_prob = float(
                self.reward_classifier.predict(images).probabilities.squeeze().cpu()
            )
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
