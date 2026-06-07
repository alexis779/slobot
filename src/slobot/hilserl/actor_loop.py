"""Actor control-loop pacing with leader GUI pumping during fps waits."""

from __future__ import annotations

import inspect
import logging

from lerobot.processor import TransitionKey
from lerobot.rl import actor as lerobot_actor
from lerobot.rl import gym_manipulator
from lerobot.utils import robot_utils

from slobot.hilserl.hilserl_gui_state import HilSerlGuiMode
from slobot.hilserl.robot_env_reset import wait_control_step
from slobot.hilserl.slobot_so100_follower import SlobotSO100Follower

_MIN_LOOP_SLEEP_S = 0.1
_ROBOT_ERRORS = (RuntimeError, TimeoutError, OSError)
logger = logging.getLogger(__name__)
_shutdown_event = None


def _disconnect_robot(env) -> None:
    robot = getattr(env, "robot", None)
    if robot is None:
        return
    try:
        robot.disconnect()
    except Exception as exc:
        logger.warning("Error disconnecting robot after hardware failure: %s", exc)


def _wait_for_recover(env, leader) -> bool:
    """Pump GUI until operator reconnects camera and presses q to end the episode."""
    while leader.gui_state.mode == HilSerlGuiMode.RECOVER:
        if _shutdown_event is not None and _shutdown_event.is_set():
            return False
        leader.pump_gui()
        leader.get_teleop_events()
        if leader.consume_recover_end_episode_request():
            robot = getattr(env, "robot", None)
            if not isinstance(robot, SlobotSO100Follower):
                logger.error("Cannot reconnect: env.robot is not SlobotSO100Follower")
                leader.enter_recover()
                continue
            try:
                robot.reconnect()
                logger.info("Robot reconnected after recover; ending episode")
                return True
            except Exception as exc:
                logger.error("Robot reconnect failed: %s", exc)
                leader.enter_recover()
                leader._set_status(
                    "Reconnect failed — fix USB cable, then press q again",
                    fg="#cf222e",
                )
        robot_utils.precise_sleep(0.05)
    return True


def _transition_after_recover(transition):
    """Mark the current episode done after a camera recover instead of resuming mid-step."""
    transition = transition.copy()
    transition[TransitionKey.DONE] = True
    transition[TransitionKey.TRUNCATED] = False
    return transition


def install_actor_recover_loop(get_leader_teleop) -> None:
    """Catch robot/camera errors, disconnect follower, and wait in recover GUI state."""
    orig_step = gym_manipulator.step_env_and_process_transition
    orig_reset = gym_manipulator.reset_and_build_transition

    def step_env_and_process_transition(
        env,
        transition,
        action,
        env_processor,
        action_processor,
    ):
        leader = get_leader_teleop()
        try:
            return orig_step(env, transition, action, env_processor, action_processor)
        except _ROBOT_ERRORS as exc:
            if leader is None:
                raise
            logger.warning("Robot hardware error, entering recover mode: %s", exc)
            _disconnect_robot(env)
            leader.enter_recover()
            if not _wait_for_recover(env, leader):
                raise
            return _transition_after_recover(transition)

    def reset_and_build_transition(env, env_processor, action_processor):
        leader = get_leader_teleop()
        try:
            return orig_reset(env, env_processor, action_processor)
        except _ROBOT_ERRORS as exc:
            if leader is None:
                raise
            logger.warning("Robot hardware error during reset, entering recover mode: %s", exc)
            _disconnect_robot(env)
            leader.enter_recover()
            if not _wait_for_recover(env, leader):
                raise
            return orig_reset(env, env_processor, action_processor)

    gym_manipulator.step_env_and_process_transition = step_env_and_process_transition
    gym_manipulator.reset_and_build_transition = reset_and_build_transition
    lerobot_actor.step_env_and_process_transition = step_env_and_process_transition
    lerobot_actor.reset_and_build_transition = reset_and_build_transition


def install_paced_act_with_policy(get_leader_teleop) -> None:
    """Patch LeRobot actor sleep so control steps wait 1/fps and pump the leader GUI."""
    orig_act = lerobot_actor.act_with_policy
    real_sleep = robot_utils.precise_sleep

    def paced_sleep(seconds: float) -> None:
        if seconds >= _MIN_LOOP_SLEEP_S:
            caller = inspect.currentframe().f_back
            if caller is not None and caller.f_code.co_name == "act_with_policy":
                cfg = caller.f_locals.get("cfg")
                start_time = caller.f_locals.get("start_time")
                if cfg is not None and cfg.env.fps is not None and start_time is not None:
                    wait_control_step(start_time, cfg.env.fps, get_leader_teleop())
                    return
        real_sleep(seconds)

    def act_with_policy(
        cfg,
        shutdown_event,
        parameters_queue,
        transitions_queue,
        interactions_queue,
    ):
        global _shutdown_event
        robot_utils.precise_sleep = paced_sleep
        lerobot_actor.precise_sleep = paced_sleep
        _shutdown_event = shutdown_event
        try:
            return orig_act(
                cfg,
                shutdown_event,
                parameters_queue,
                transitions_queue,
                interactions_queue,
            )
        finally:
            _shutdown_event = None
            robot_utils.precise_sleep = real_sleep
            lerobot_actor.precise_sleep = real_sleep

    lerobot_actor.act_with_policy = act_with_policy
    logging.debug("Installed paced act_with_policy (env.fps honored with leader GUI pump)")
