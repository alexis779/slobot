"""HIL-SERL processor pipelines (leader teleop + gripper_link EE pose)."""

from __future__ import annotations

import gymnasium as gym

from lerobot.processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    RewardClassifierProcessorStep,
    VanillaObservationProcessorStep,
    identity_transition,
)
from lerobot.rl.gym_manipulator import make_processors as _orig_make_processors
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEEObservation

from slobot.hilserl.ee_action_processors import (
    EeActionRecordProcessorStep,
    EeActionToJointProcessorStep,
    LeaderTeleopToEeActionProcessorStep,
    PolicyEeActionProcessorStep,
    SimCollisionGateProcessorStep,
)
from slobot.hilserl.ee_pose_state_processor import (
    EEPoseInObservationStateStep,
    set_kinematics_cfg,
)
from slobot.hilserl.factory import Factory
from slobot.hilserl.leader_intervention_processor import LeaderInterventionActionProcessorStep
from slobot.hilserl.leader_joint_processor import LeaderJointTeleopProcessorStep
from slobot.hilserl.reward_classifier_processor import SlobotRewardClassifierProcessorStep
from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop
from slobot.hilserl.teleop_gripper_processor import TeleopGripperCommandProcessorStep


def _configure_env_processor(
    env_processor: DataProcessorPipeline, cfg
) -> DataProcessorPipeline:
    """Drop LeRobot joint FK; use Genesis gripper_link FK for observation.state."""
    steps = [
        step
        for step in env_processor.steps
        if not isinstance(step, ForwardKinematicsJointsToEEObservation)
    ]
    obs_cfg = cfg.processor.observation
    if obs_cfg is not None and obs_cfg.add_ee_pose_to_observation:
        insert_at = next(
            (i + 1 for i, step in enumerate(steps) if isinstance(step, VanillaObservationProcessorStep)),
            len(steps),
        )
        steps.insert(insert_at, EEPoseInObservationStateStep())
    return DataProcessorPipeline(
        steps=steps,
        to_transition=env_processor.to_transition,
        to_output=env_processor.to_output,
    )


def _use_slobot_reward_classifier(
    env_processor: DataProcessorPipeline,
    cfg,
) -> DataProcessorPipeline:
    steps = list(env_processor.steps)
    for i, step in enumerate(steps):
        if isinstance(step, RewardClassifierProcessorStep) and not isinstance(
            step, SlobotRewardClassifierProcessorStep
        ):
            steps[i] = SlobotRewardClassifierProcessorStep(
                pretrained_path=step.pretrained_path,
                device=step.device,
                success_threshold=step.success_threshold,
                success_reward=step.success_reward,
                terminate_on_success=step.terminate_on_success,
            )
            break
    return DataProcessorPipeline(
        steps=steps,
        to_transition=env_processor.to_transition,
        to_output=env_processor.to_output,
    )


def _max_ee_step_m(ik_cfg, fps: float) -> float:
    sizes = ik_cfg.end_effector_step_sizes
    base = max(
        float(sizes.get("x", 0.005)),
        float(sizes.get("y", 0.005)),
        float(sizes.get("z", 0.005)),
    )
    return base * min(max(fps, 1.0), 10.0)


def _collision_penalty(cfg) -> float:
    kin = getattr(cfg.processor, "kinematics", None)
    if kin is not None and hasattr(kin, "collision_penalty"):
        return float(kin.collision_penalty)
    return -1.0


def _build_action_processor(
    env: gym.Env,
    teleop_device: SlobotSO100LeaderTeleop,
    cfg,
) -> DataProcessorPipeline:
    if cfg.processor.inverse_kinematics is None:
        raise RuntimeError("HIL-SERL requires processor.inverse_kinematics")

    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )
    ik_cfg = cfg.processor.inverse_kinematics
    motor_names = list(env.robot.bus.motors.keys())
    bundle = Factory.get_kinematics_bundle(
        gripper_link_name=ik_cfg.gripper_link_name,
        jaw_joint_name=ik_cfg.jaw_joint_name,
        motor_names=motor_names,
        fps=cfg.fps,
    )

    steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        TeleopGripperCommandProcessorStep(teleop_device=teleop_device),
        LeaderJointTeleopProcessorStep(motor_names=motor_names),
        LeaderInterventionActionProcessorStep(
            motor_names=motor_names,
            terminate_on_success=terminate_on_success,
        ),
        LeaderTeleopToEeActionProcessorStep(motor_names=motor_names, bundle=bundle),
        PolicyEeActionProcessorStep(
            motor_names=motor_names,
            max_ee_step_m=_max_ee_step_m(ik_cfg, cfg.fps),
            jaw_limits=bundle.jaw_limits,
        ),
        SimCollisionGateProcessorStep(
            bundle=bundle,
            motor_names=motor_names,
            collision_penalty=_collision_penalty(cfg),
        ),
        EeActionToJointProcessorStep(motor_names=motor_names),
        EeActionRecordProcessorStep(),
    ]
    return DataProcessorPipeline(
        steps=steps, to_transition=identity_transition, to_output=identity_transition
    )


def _require_leader_teleop(teleop_device) -> SlobotSO100LeaderTeleop:
    if teleop_device is None:
        raise RuntimeError("HIL-SERL requires a leader teleop device (cfg.teleop)")
    if not isinstance(teleop_device, SlobotSO100LeaderTeleop):
        raise TypeError(
            f"HIL-SERL requires slobot_so100_leader teleop, got {type(teleop_device).__name__}"
        )
    return teleop_device


def make_hilserl_processors(
    env: gym.Env,
    teleop_device,
    cfg,
    device: str = "cpu",
) -> tuple[DataProcessorPipeline, DataProcessorPipeline]:
    leader = _require_leader_teleop(teleop_device)
    motor_names = list(env.robot.bus.motors.keys())
    set_kinematics_cfg(cfg, motor_names=motor_names)

    # HIL-SERL uses Genesis IK; skip LeRobot's default inverse-kinematics processor steps.
    ik_cfg = cfg.processor.inverse_kinematics
    cfg.processor.inverse_kinematics = None
    try:
        env_processor, _ = _orig_make_processors(env, leader, cfg, device)
    finally:
        cfg.processor.inverse_kinematics = ik_cfg
    env_processor = _configure_env_processor(env_processor, cfg)
    env_processor = _use_slobot_reward_classifier(env_processor, cfg)
    action_processor = _build_action_processor(env, leader, cfg)
    return env_processor, action_processor
