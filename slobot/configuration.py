import numpy as np

class Configuration:
    QPOS_MAP = {
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "zero": [0, 0, 0, 0, 0, 0],
        "rest": [0.049, -3.62, 3.19, 1.26, -0.17, -0.67]
    }

    POS_MAP = {
        "rotated": [3147, 2061, 2001, 3017, 1121, 3438],
        "zero": [2112, 3069, 996, 2099, 2111, 2088],
        "rest": [2091, 728, 3056, 2812, 2032, 1977]
    }