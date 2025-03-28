import numpy as np
import logging

class Configuration:
    WORK_DIR = "/tmp/slobot"

    MJCF_CONFIG = './trs_so_arm100/so_arm100.xml'

    # 16:9 aspect ratio
    LD = (426, 240)
    SD = (854, 480)
    HD = (1280, 720)
    FHD = (1920, 1080)

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

    def logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger