from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PreGraspMode(Enum):
    VERTICAL = "vertical"
    VERTICAL_FLIP = "vertical-flip"
    HORIZONTAL = "horizontal"

@dataclass
class RecordingLayout:
    rrd_file: Optional[str]
    pick_frame_id: Optional[int]
    pre_grasp_mode: Optional[PreGraspMode]
    ball_x: float
    ball_y: float
    cup_x: float
    cup_y: float
    recording_id: str # the destination recording id