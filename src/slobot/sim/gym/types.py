from __future__ import annotations

from typing import TypedDict

import numpy as np


class GolfBallEnvObservation(TypedDict):
    qpos: np.ndarray
    side_camera_image: np.ndarray
    link_camera_image: np.ndarray


class GolfBallEnvAction(TypedDict):
    control_qpos: np.ndarray
