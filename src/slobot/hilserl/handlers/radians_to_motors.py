"""Convert joint radians to motor steps."""

from __future__ import annotations

from slobot.feetech import Feetech
from slobot.hilserl.models.motor_io import MotorRadians, MotorSteps


class RadiansToMotors:
    def __init__(self, feetech: Feetech) -> None:
        self._feetech = feetech

    def convert(self, radians: MotorRadians) -> MotorSteps:
        pos = self._feetech.qpos_to_pos(radians.to_list())
        return MotorSteps.from_list(pos)
