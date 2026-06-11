"""Run gym_manipulator with slobot HIL-SERL plugins (follower robot, leader teleop)."""

from __future__ import annotations

from slobot.hilserl.gui_init import init_x11_threading

init_x11_threading()

import logging
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

# Registers HIL-SERL plugins before LeRobot imports (robot type, teleop, processor).
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config

register_hilserl_processor_config()

from lerobot.datasets import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.rl import gym_manipulator
from lerobot.rl.gym_manipulator import main
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, HF_LEROBOT_HOME, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from slobot.hilserl.gui_init import make_robot_env_with_genesis_before_tk
from slobot.hilserl.hilserl_gui_state import HilSerlGuiContext
from slobot.hilserl.hilserl_processors import make_hilserl_processors
from slobot.hilserl.factory import Factory
from slobot.hilserl.ee_action_processors import EE_RECORD_ACTION_FEATURES
from slobot.hilserl.reward_classifier_processor import (
    REWARD_PROBABILITY,
    find_reward_classifier_step,
)
from slobot.hilserl.robot_env_reset import robot_env_reset as perform_robot_env_reset
from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop
from slobot.hilserl.dataset_image_keys import (
    camera_name_from_video_key,
    is_video_image_key,
    preprocessed_camera_key,
)
Factory.install()

_orig_make_robot_env = gym_manipulator.make_robot_env
_orig_reset_and_build_transition = gym_manipulator.reset_and_build_transition
_orig_control_loop = gym_manipulator.control_loop
_orig_save_episode = LeRobotDataset.save_episode
_leader_teleop: SlobotSO100LeaderTeleop | None = None
_num_episodes: int | None = None
_saved_episodes = 0
_last_transition = None


def robot_env_reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
):
    if _leader_teleop is None:
        raise RuntimeError("Leader teleop not initialized; call make_robot_env first")
    return perform_robot_env_reset(self, _leader_teleop, seed=seed, options=options)


def make_robot_env(cfg):
    global _leader_teleop
    if cfg.name == "gym_hil":
        env, teleop_device = _orig_make_robot_env(cfg)
    else:
        env, teleop_device = make_robot_env_with_genesis_before_tk(cfg, factory=Factory)
    if not isinstance(teleop_device, SlobotSO100LeaderTeleop):
        raise TypeError(
            f"HIL-SERL record requires slobot_so100_leader teleop, got {type(teleop_device).__name__}"
        )
    _leader_teleop = teleop_device
    teleop_device.set_control_fps(cfg.fps)
    teleop_device.set_gui_context(HilSerlGuiContext.RECORD)
    env.leader_teleop = teleop_device
    if cfg.name != "gym_hil" and cfg.processor.reset is not None:
        env.reset_time_s = cfg.processor.reset.reset_time_s
    return env, teleop_device


def reset_and_build_transition(env, env_processor, action_processor):
    global _last_transition
    if _num_episodes is not None and _saved_episodes >= _num_episodes:
        logging.info("Recording complete, skipping environment reset.")
        log_say("Recording complete.", play_sounds=True)
        _leader_teleop.notify_recording_complete()
        return _last_transition

    _leader_teleop.notify_episode(_saved_episodes, _num_episodes)
    _leader_teleop.notify_resetting()
    transition = _orig_reset_and_build_transition(env, env_processor, action_processor)
    _last_transition = transition
    log_say("Start teleoperating.", play_sounds=True)
    _leader_teleop.notify_recording_started()
    return transition


def _tracked_save_episode(self, *args, **kwargs):
    global _saved_episodes
    result = _orig_save_episode(self, *args, **kwargs)
    _saved_episodes += 1
    return result


def control_loop(env, env_processor, action_processor, teleop_device, cfg):
    global _num_episodes, _saved_episodes, _last_transition
    if cfg.mode == "record":
        _num_episodes = cfg.dataset.num_episodes_to_record
        _saved_episodes = 0
        _last_transition = None
    return _record_control_loop(env, env_processor, action_processor, teleop_device, cfg)


def _env_has_reward_classifier(env_cfg) -> bool:
    reward_classifier = getattr(env_cfg.processor, "reward_classifier", None)
    return reward_classifier is not None and reward_classifier.pretrained_path is not None


def _preprocessed_feature_spec(shape: tuple[int, ...]) -> dict:
    """Parquet feature for live-resized float32 [C, H, W] classifier tensors."""
    return {
        "dtype": "float32",
        "shape": shape,
        "names": ["channels", "height", "width"],
    }


def _recording_dataset_root(cfg) -> Path:
    if cfg.dataset.root is not None:
        return Path(cfg.dataset.root)
    return HF_LEROBOT_HOME / cfg.dataset.repo_id


def _open_recording_dataset(
    cfg,
    features: dict,
) -> LeRobotDataset:
    root = _recording_dataset_root(cfg)
    writer_kwargs = {
        "image_writer_threads": 4,
        "image_writer_processes": 0,
    }
    if (root / "meta" / "info.json").exists():
        logging.info("Resuming existing dataset at %s", root)
        return LeRobotDataset.resume(cfg.dataset.repo_id, root=root, **writer_kwargs)

    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.env.fps,
        root=cfg.dataset.root,
        use_videos=True,
        features=features,
        **writer_kwargs,
    )


def _observation_for_frame(observation: dict[str, Any]) -> dict[str, Any]:
    """Build the frame dict: video keys for AV1, preprocessed keys for parquet."""
    frame: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        if is_video_image_key(key):
            camera = camera_name_from_video_key(key)
            if camera is not None:
                frame[preprocessed_camera_key(camera)] = np.asarray(value, dtype=np.float32)
            frame[key] = value
        else:
            frame[key] = value
    return frame


def _record_control_loop(env, env_processor, action_processor, teleop_device, cfg):
    """Record gripper_link EE pose + gripper command actions from leader teleop."""
    dt = 1.0 / cfg.env.fps
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    record_reward_prob = _env_has_reward_classifier(cfg.env)
    reward_classifier_step = (
        find_reward_classifier_step(env_processor) if record_reward_prob else None
    )
    # Store live resized tensors in parquet (observation.preprocessed.*) alongside AV1 video.
    transition = reset_and_build_transition(env, env_processor, action_processor)

    dataset = None
    if cfg.mode == "record":
        action_features = EE_RECORD_ACTION_FEATURES
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }
        if record_reward_prob:
            features[REWARD_PROBABILITY] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": list(EE_RECORD_ACTION_FEATURES["names"]),
                }
            if is_video_image_key(key):
                shape = tuple(value.squeeze(0).shape)
                features[key] = {
                    "dtype": "video",
                    "shape": shape,
                    "names": ["channels", "height", "width"],
                }
                camera = camera_name_from_video_key(key)
                if camera is not None:
                    features[preprocessed_camera_key(camera)] = _preprocessed_feature_spec(shape)

        dataset = _open_recording_dataset(cfg, features)

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            neutral_action = torch.tensor([0.0] * 6, dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])

            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            frame_observation = _observation_for_frame(observation)

            transition = gym_manipulator.step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            if cfg.mode == "record":
                complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                action_to_record = complementary.get("teleop_action", transition[TransitionKey.ACTION])
                frame = {
                    **frame_observation,
                    ACTION: action_to_record.cpu()
                    if hasattr(action_to_record, "cpu")
                    else action_to_record,
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = complementary.get("discrete_penalty", 0.0)
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                reward_prob: float | None = None
                if record_reward_prob:
                    reward_prob = 0.0
                    if reward_classifier_step is not None:
                        predicted = reward_classifier_step.predict_probability_from_observation(
                            observation
                        )
                        if predicted is not None:
                            reward_prob = predicted
                    frame[REWARD_PROBABILITY] = np.array([reward_prob], dtype=np.float32)

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)

            episode_step += 1

            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s "
                    f"with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                transition = reset_and_build_transition(env, env_processor, action_processor)

            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
    finally:
        if dataset is not None and dataset.writer is not None and dataset.writer.image_writer is not None:
            logging.info("Waiting for image writer to finish...")
            dataset.writer.image_writer.stop()

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Finalizing dataset before pushing to hub")
        dataset.finalize()
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


gym_manipulator.make_processors = make_hilserl_processors
gym_manipulator.make_robot_env = make_robot_env
gym_manipulator.reset_and_build_transition = reset_and_build_transition
gym_manipulator.control_loop = control_loop
gym_manipulator.RobotEnv.reset = robot_env_reset
LeRobotDataset.save_episode = _tracked_save_episode
