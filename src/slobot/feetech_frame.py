from dataclasses import dataclass

@dataclass
class FeetechFrame:
    timestamp: float = None
    control_pos: list[float] = None
    qpos: list[float] = None
    velocity: list[float] = None
    control_force: list[float] = None