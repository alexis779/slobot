"""Run HIL-SERL learner with slobot plugins (custom robot and teleop configs)."""

from __future__ import annotations

import logging
from typing import Any

# Register custom robot, teleop, and processor configs before LeRobot parses CLI args.
import slobot.hilserl.config_slobot_so100_follower  # noqa: F401
import slobot.hilserl.config_slobot_so100_leader  # noqa: F401
from slobot.hilserl.config_hilserl_env import register_hilserl_processor_config

register_hilserl_processor_config()

from torch.multiprocessing import Queue

from lerobot.common.wandb_utils import WandBLogger
from lerobot.rl import learner as lerobot_learner
from lerobot.rl.learner import (
    add_actor_information_and_train,
    start_learner,
    train_cli,
    use_threads,
)
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

from slobot.hilserl.logging_utils import patch_init_logging_for_tracebacks

patch_init_logging_for_tracebacks(lerobot_learner)


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: Any,
) -> None:
    """Like LeRobot's start_learner_threads, but logs full tracebacks and re-raises."""
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    try:
        add_actor_information_and_train(
            cfg=cfg,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
            transition_queue=transition_queue,
            interaction_message_queue=interaction_message_queue,
            parameters_queue=parameters_queue,
        )
        logging.info("[LEARNER] Training process stopped")
    except Exception:
        logging.exception("[LEARNER] Unhandled exception in training loop")
        shutdown_event.set()
        raise
    finally:
        logging.info("[LEARNER] Closing queues")
        transition_queue.close()
        interaction_message_queue.close()
        parameters_queue.close()

        communication_process.join()
        logging.info("[LEARNER] Communication process joined")

        transition_queue.cancel_join_thread()
        interaction_message_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

        logging.info("[LEARNER] Cleanup complete")


lerobot_learner.start_learner_threads = start_learner_threads
