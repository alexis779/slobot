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


@dataclass
@ProcessorStepRegistry.register("slobot_reward_classifier_processor")
class SlobotRewardClassifierProcessorStep(RewardClassifierProcessorStep):
    """Applies classifier_preprocessor.json before predict_reward (LeRobot omits this)."""

    preprocessor: PolicyProcessorPipeline | None = field(default=None, repr=False)
    min_steps_after_reset: int = 20
    min_consecutive_success_frames: int = 1
    _steps_since_reset: int = field(default=0, init=False, repr=False)
    _consecutive_success_frames: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._steps_since_reset = 0
        self._consecutive_success_frames = 0

    def _should_terminate_on_classifier_success(self, transition: EnvTransition) -> bool:
        if not self.terminate_on_success:
            return False
        if self._steps_since_reset < self.min_steps_after_reset:
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

        batch = {
            key: observation[key]
            for key in self.reward_classifier.config.input_features
            if key in observation
        }
        if len(batch) != len(self.reward_classifier.config.input_features):
            return new_transition

        self._steps_since_reset += 1

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

        if success:
            self._consecutive_success_frames += 1
            reward = self.success_reward
            if (
                self._consecutive_success_frames >= self.min_consecutive_success_frames
                and self._should_terminate_on_classifier_success(new_transition)
            ):
                terminated = True
        else:
            self._consecutive_success_frames = 0

        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = 1 / (time.perf_counter() - start_time)
        info[REWARD_PROBABILITY] = success_prob
        new_transition[TransitionKey.INFO] = info

        if success_prob > 0.1:
            logging.info(
                "Reward classifier prob=%.4f (threshold=%.2f)",
                success_prob,
                self.success_threshold,
            )

        return new_transition
