"""Lazy singletons and LeRobot wiring for HIL-SERL hardware and sim."""

from __future__ import annotations

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config as _lerobot_make_robot_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import make_teleoperator_from_config as _lerobot_make_teleoperator_from_config

from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.hilserl.config_slobot_so100_follower import SlobotSO100FollowerConfig
from slobot.hilserl.config_slobot_so100_leader import SlobotSO100LeaderTeleopConfig
from slobot.hilserl.corner_env import CornerEnv
from slobot.hilserl.handlers.collision_handler import GenesisCollisionHandler
from slobot.hilserl.handlers.fk_handler import FkHandler
from slobot.hilserl.handlers.ik_handler import IkHandler
from slobot.hilserl.handlers.motor_qpos import (
    apply_gripper_to_joint_targets,
    jaw_joint_limits,
    resolve_jaw_indices,
)
from slobot.hilserl.handlers.motors_to_radians import MotorsToRadians
from slobot.hilserl.handlers.radians_to_motors import RadiansToMotors
from slobot.hilserl.kinematics_bundle import KinematicsBundle
from slobot.hilserl.models.gripper_command import GripperCommand
from slobot.robotic_arm import RoboticArm


class Factory:
    _robotic_arm: RoboticArm | None = None
    _follower_feetech: Feetech | None = None
    _leader_feetech: Feetech | None = None
    _motors_to_radians: MotorsToRadians | None = None
    _radians_to_motors: RadiansToMotors | None = None
    _kinematics_bundle: KinematicsBundle | None = None
    _installed = False

    @staticmethod
    def get_robotic_arm(**kwargs) -> RoboticArm:
        if Factory._robotic_arm is None:
            defaults = {
                "show_viewer": True,
                "mjcf_path": "../mujoco_menagerie/trs_so_arm100/so_arm100.xml",
                "should_start": False,
                "fps": 10,
            }
            defaults.update(kwargs)
            Factory._robotic_arm = RoboticArm(**defaults)
            CornerEnv(Factory._robotic_arm)
        return Factory._robotic_arm

    @staticmethod
    def get_follower_feetech(**kwargs) -> Feetech:
        if Factory._follower_feetech is None:
            defaults = {
                "connect": True,
                "qpos_map": Configuration.MJCF_QPOS_MAP,
                "port": Feetech.PORT_FOLLOWER,
                "robot_id": Feetech.FOLLOWER_ID,
            }
            defaults.update(kwargs)
            if "qpos_handler" not in defaults:
                defaults["qpos_handler"] = Factory.get_robotic_arm()
            Factory._follower_feetech = Feetech(**defaults)
        return Factory._follower_feetech

    @staticmethod
    def get_leader_feetech(**kwargs) -> Feetech:
        if Factory._leader_feetech is None:
            defaults = {
                "connect": True,
                "qpos_map": Configuration.MJCF_QPOS_MAP,
                "port": Feetech.PORT_LEADER,
                "robot_id": Feetech.LEADER_ID,
            }
            defaults.update(kwargs)
            Factory._leader_feetech = Feetech(**defaults)
        return Factory._leader_feetech

    @staticmethod
    def get_motors_to_radians() -> MotorsToRadians:
        if Factory._motors_to_radians is None:
            Factory._motors_to_radians = MotorsToRadians(Factory.get_follower_feetech())
        return Factory._motors_to_radians

    @staticmethod
    def get_radians_to_motors() -> RadiansToMotors:
        if Factory._radians_to_motors is None:
            Factory._radians_to_motors = RadiansToMotors(Factory.get_follower_feetech())
        return Factory._radians_to_motors

    @staticmethod
    def get_kinematics_bundle(
        *,
        gripper_link_name: str,
        jaw_joint_name: str,
        motor_names: list[str],
        fps: int,
    ) -> KinematicsBundle:
        if Factory._kinematics_bundle is not None:
            return Factory._kinematics_bundle

        robotic_arm = Factory.get_robotic_arm()
        n_dofs = robotic_arm.genesis.entity.n_dofs
        jaw_joint_idx, jaw_motor_idx = resolve_jaw_indices(
            robotic_arm, jaw_joint_name, motor_names
        )
        jaw_limits = jaw_joint_limits(robotic_arm, jaw_joint_name)

        fk = FkHandler(
            robotic_arm,
            n_dofs=n_dofs,
            jaw_joint_idx=jaw_joint_idx,
            gripper_link_name=gripper_link_name,
            jaw_joint_name=jaw_joint_name,
        )
        ik = IkHandler(
            robotic_arm,
            n_dofs=n_dofs,
            n_motors=len(motor_names),
            jaw_joint_idx=jaw_joint_idx,
            gripper_link_name=gripper_link_name,
            jaw_joint_name=jaw_joint_name,
        )
        collision = GenesisCollisionHandler(robotic_arm, n_dofs=n_dofs)

        Factory._kinematics_bundle = KinematicsBundle(
            fk=fk,
            ik=ik,
            collision=collision,
            n_dofs=n_dofs,
            n_motors=len(motor_names),
            jaw_joint_idx=jaw_joint_idx,
            jaw_motor_idx=jaw_motor_idx,
            jaw_limits=jaw_limits,
            gripper_link_name=gripper_link_name,
            jaw_joint_name=jaw_joint_name,
        )
        return Factory._kinematics_bundle

    @staticmethod
    def reset() -> None:
        for feetech in (Factory._follower_feetech, Factory._leader_feetech):
            if feetech is not None:
                try:
                    feetech.disconnect()
                except Exception:
                    pass
        Factory._follower_feetech = None
        Factory._leader_feetech = None
        Factory._robotic_arm = None
        Factory._motors_to_radians = None
        Factory._radians_to_motors = None
        Factory._kinematics_bundle = None

    @staticmethod
    def make_robot_from_config(config: RobotConfig) -> Robot:
        if isinstance(config, SlobotSO100FollowerConfig):
            from slobot.hilserl.slobot_so100_follower import SlobotSO100Follower

            return SlobotSO100Follower(config)
        return _lerobot_make_robot_from_config(config)

    @staticmethod
    def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
        if isinstance(config, SlobotSO100LeaderTeleopConfig):
            from slobot.hilserl.slobot_so100_leader import SlobotSO100LeaderTeleop

            return SlobotSO100LeaderTeleop(config)
        return _lerobot_make_teleoperator_from_config(config)

    @staticmethod
    def _patch_robot_env() -> None:
        import lerobot.rl.gym_manipulator as gym_manipulator
        from lerobot.teleoperators.utils import TeleopEvents

        from slobot.hilserl.slobot_so100_follower import SlobotSO100Follower

        if getattr(gym_manipulator.RobotEnv, "_slobot_patched", False):
            return

        orig_step = gym_manipulator.RobotEnv.step
        motor_names = list(Configuration.JOINT_NAMES)

        def _get_observation(self):
            obs_dict = self.robot.get_observation()
            joint_positions = np.array([obs_dict[f"{name}.pos"] for name in motor_names])
            image_keys = [key for key in obs_dict if key not in {f"{name}.pos" for name in motor_names}]
            images = {key: obs_dict[key] for key in image_keys}
            return {"agent_pos": joint_positions, "pixels": images, **obs_dict}

        def step(self, action):
            if isinstance(self.robot, SlobotSO100Follower):
                joint_targets_dict = {
                    f"{name}.pos": int(action[i]) for i, name in enumerate(self.robot.motor_names)
                }
                bundle = Factory._kinematics_bundle
                feetech = self.robot._feetech

                previous_gripper_step = None
                if self._raw_joint_positions is not None:
                    gripper_step = self._raw_joint_positions.get("gripper.pos")
                    if gripper_step is not None:
                        previous_gripper_step = float(gripper_step)

                gripper_command = GripperCommand.STAY
                leader_teleop = getattr(self, "leader_teleop", None)
                if leader_teleop is not None and hasattr(leader_teleop, "gripper_command"):
                    gripper_command = leader_teleop.gripper_command

                if bundle is not None and leader_teleop is not None:
                    apply_gripper_to_joint_targets(
                        joint_targets_dict,
                        command=gripper_command,
                        jaw_limits=bundle.jaw_limits,
                        feetech=feetech,
                        jaw_motor_idx=bundle.jaw_motor_idx,
                        motor_names=motor_names,
                        previous_gripper_step=previous_gripper_step,
                    )

                self.robot.send_action(joint_targets_dict)

                obs = _get_observation(self)
                self._raw_joint_positions = {
                    f"{key}.pos": obs[f"{key}.pos"] for key in motor_names
                }
                if self.display_cameras:
                    self.render()
                self.current_step += 1
                return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}
            return orig_step(self, action)

        gym_manipulator.RobotEnv._get_observation = _get_observation
        gym_manipulator.RobotEnv.step = step
        gym_manipulator.RobotEnv._slobot_patched = True

    @staticmethod
    def install() -> None:
        """Patch LeRobot so slobot followers receive the shared Feetech instance."""
        if Factory._installed:
            return
        import lerobot.rl.gym_manipulator as gym_manipulator
        import lerobot.robots.utils as robots_utils
        import lerobot.teleoperators.utils as teleoperators_utils

        robots_utils.make_robot_from_config = Factory.make_robot_from_config
        teleoperators_utils.make_teleoperator_from_config = Factory.make_teleoperator_from_config
        gym_manipulator.make_robot_from_config = Factory.make_robot_from_config
        Factory._patch_robot_env()
        Factory._installed = True
