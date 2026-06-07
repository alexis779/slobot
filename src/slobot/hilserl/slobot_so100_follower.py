"""SO-100 follower with slobot joint observations for HIL-SERL."""

from __future__ import annotations

import logging
import time

from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from slobot.feetech import Feetech
from slobot.hilserl.config_slobot_so100_follower import SlobotSO100FollowerConfig
from slobot.hilserl.factory import Factory

logger = logging.getLogger(__name__)


class SlobotSO100Follower(Robot):
    """Follower that reports raw motor steps and camera frames via Feetech."""

    POS_CHANGE_THRESHOLD = 0.75
    MAX_CAM_AGE_MS = 1000

    config_class = SlobotSO100FollowerConfig
    name = "slobot_so100_follower"

    def __init__(self, config: SlobotSO100FollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self._feetech = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def bus(self):
        return self._feetech.motors_bus

    @property
    def motor_names(self) -> list[str]:
        return list(self._feetech.motors_bus.motors.keys())

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        motors = {f"{motor}.pos": int for motor in self.motor_names}
        cameras = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        return {**motors, **cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": int for motor in self.motor_names}

    @property
    def is_connected(self) -> bool:
        feetech_connected = self._feetech is not None and self._feetech.motors_bus.is_connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return feetech_connected and cameras_connected

    @property
    def is_calibrated(self) -> bool:
        return self._feetech.motors_bus.is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._feetech = Factory.get_follower_feetech(
            port=self.config.port,
            robot_id=self.config.id,
            torque=True,
            qpos_handler=Factory.get_robotic_arm() if self.cameras else None,
        )
        for cam in self.cameras.values():
            cam.connect()
        logger.info("%s connected.", self)

    def reconnect(self) -> None:
        """Disconnect and connect again after a hardware error (e.g. camera USB drop)."""
        try:
            self.disconnect()
        except Exception:
            pass
        Factory._follower_feetech = None
        self._feetech = None
        self.connect()

    @check_if_not_connected
    def get_observation(self) -> dict:
        start = time.perf_counter()
        self.previous_pos = self._feetech.get_pos()
        observation = {
            f"{name}.pos": self.previous_pos[i] for i, name in enumerate(self.motor_names)
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            observation[cam_key] = cam.read_latest(max_age_ms=self.MAX_CAM_AGE_MS)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return observation

    @check_if_not_connected
    def send_action(self, action: dict[str, int]) -> dict[str, int]:
        pos = [action[f"{name}.pos"] for name in self.motor_names]
        self._safe_action(pos)
        self._feetech.control_position(pos)
        return action

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        if self._feetech is not None:
            try:
                self._feetech.disconnect()
            except Exception:
                pass
        logger.info("%s disconnected.", self)

    def _safe_action(self, pos: list[int]) -> None:
        for i in range(len(self.motor_names)):
            if abs(pos[i] - self.previous_pos[i])/self._feetech.model_resolution > self.POS_CHANGE_THRESHOLD:
                logger.warning(f"{self} motor {self.motor_names[i]} position change unsafe {self.previous_pos[i]} -> {pos[i]}")
                pos[i] = self.previous_pos[i]