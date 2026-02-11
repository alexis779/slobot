import numpy as np
import logging
from importlib.resources import files

class Configuration:
    WORK_DIR = "/tmp/slobot"

    # 4:3 aspect ratio
    QVGA = (320, 240)
    VGA = (640, 480)
    XGA = (1024, 768)
    UXGA = (1600, 1200)

    QPOS_MAP = {
        "middle": [0, -np.pi/2, np.pi/2, 0, 0, -0.15],
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17]
    }

    POS_MAP = {
        "middle": [2047, 2047, 2047, 2047, 2047, 2047],
        "zero": [2047, 3083, 1030, 2048, 2047, 2144],
        "rotated": [3071, 2052, 2051, 3071, 1023, 3168],
        "rest": [2016, 907, 3070, 2831, 1937, 2035],
    }

    REFERENCE_FRAME = 'middle'

    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    GRIPPER_ID = 5 # the id of the jaw joint

    def logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False # avoid propagating to the root logger, otherwise each log line shows twice in stderr
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger