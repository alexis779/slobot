from dataclasses import dataclass
from enum import Enum


class PreGraspMode(Enum):
    VERTICAL = "vertical"
    VERTICAL_FLIP = "vertical-flip"
    HORIZONTAL = "horizontal"

@dataclass
class RecordingLayout:
    rrd_file: str
    pick_frame_id: int
    pre_grasp_mode: PreGraspMode
    ball_x: float
    ball_y: float
    cup_x: float
    cup_y: float