from dataclasses import dataclass
import torch

@dataclass
class ConfigurationMapping:
    episode_id: int
    qpos: torch.Tensor
    motor_pos: torch.Tensor
    link_quat: torch.Tensor