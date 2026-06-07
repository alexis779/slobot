"""Wire slobot HIL-SERL processor config into LeRobot env config parsing."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig

from slobot.hilserl.config_inverse_kinematics import SlobotGenesisConfig


@dataclass
class SlobotHILSerlProcessorConfig(HILSerlProcessorConfig):
    inverse_kinematics: SlobotGenesisConfig | None = None


def register_hilserl_processor_config() -> None:
    """Use SlobotGenesisConfig when decoding env.processor from JSON."""
    HILSerlRobotEnvConfig.__annotations__["processor"] = SlobotHILSerlProcessorConfig
    proc_field = HILSerlRobotEnvConfig.__dataclass_fields__["processor"]
    proc_field.type = SlobotHILSerlProcessorConfig
    proc_field.default_factory = SlobotHILSerlProcessorConfig

    SlobotHILSerlProcessorConfig.__annotations__["inverse_kinematics"] = (
        SlobotGenesisConfig | None
    )
    ik_field = SlobotHILSerlProcessorConfig.__dataclass_fields__["inverse_kinematics"]
    ik_field.type = SlobotGenesisConfig | None
