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
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17]
    }

    POS_MAP = {
        "zero": [2035, 3081, 1001, 1966, 1988, 2125],
        "rotated": [3052, 2021, 2040, 3062, 905, 3179],
        "rest": [2068, 819, 3051, 2830, 2026, 2049],
    }

    DOFS = 6
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger