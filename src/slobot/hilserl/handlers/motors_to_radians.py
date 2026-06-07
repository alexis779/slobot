"""Convert motor steps to joint radians."""

from __future__ import annotations

from slobot.feetech import Feetech
from slobot.hilserl.models.motor_io import MotorRadians, MotorSteps


class MotorsToRadians:
    def __init__(self, feetech: Feetech) -> None:
        self._feetech = feetech

    def convert(self, steps: MotorSteps) -> MotorRadians:
        qpos = self._feetech.pos_to_qpos(steps.to_list())
        return MotorRadians.from_list(qpos)
