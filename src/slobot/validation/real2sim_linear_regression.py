import torch
from dataclasses import dataclass

@dataclass
class RecordingPosition:
    recording_id: str
    ball_x: float
    ball_y: float
    motor_pos: torch.Tensor
    qpos: torch.Tensor

class Real2SimLinearRegression:
    def __init__(self):
        pass