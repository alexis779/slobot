import numpy as np
import logging
from importlib.resources import files

class Configuration:
    WORK_DIR = "/tmp/slobot"

    MJCF_CONFIG = str(files('slobot.config') / "trs_so_arm100" / "so_arm100.xml")

    # 16:9 aspect ratio
    LD = (426, 240)
    SD = (854, 480)
    HD = (1280, 720)
    FHD = (1920, 1080)

    QPOS_MAP = {
        "middle": [0, -np.pi/2, np.pi/2, 0, 0, -0.15],
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17]
    }

    POS_MAP = {
        "middle": [2095, 2095, 2095, 2095, 2095, 2095],
        "zero": [2097, 3143, 978, 1997, 1081, 2162],
        "rotated": [3161, 2126, 2016, 3017, 12, 3330],
        "rest": [2089, 965, 3063, 2822, 1035, 2030]
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