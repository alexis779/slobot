import torch
import logging
import os

from importlib.resources import files
import slobot.config

_CONFIG = files(slobot.config)

class Configuration:
    WORK_DIR = f"{os.environ['HOME']}/.slobot"

    # 4:3 aspect ratio
    QVGA = (320, 240)
    VGA = (640, 480)
    XGA = (1024, 768)
    UXGA = (1600, 1200)

    # MJCF
    MJCF_CONFIG = str(_CONFIG.joinpath("trs_so_arm100", "so_arm100.xml")) # "../mujoco_menagerie/trs_so_arm100/so_arm100.xml"
    MJCF_GRIPPER_LINK_NAME = 'Fixed_Jaw'
    MJCF_GRIPPER_JOINT_NAME = 'Jaw'

    # URDF
    URDF_CONFIG = str(_CONFIG.joinpath("SO100", "so100.urdf")) # "../SO-ARM100/Simulation/SO100/so100.urdf"
    URDF_GRIPPER_LINK_NAME = 'gripper_link'
    URDF_GRIPPER_JOINT_NAME = 'gripper'

    MJCF_QPOS_MAP = {
        "middle": [0, -torch.pi/2, torch.pi/2, 0, 0, -0.15],
        "zero": [0, 0, 0, 0, 0, 0],
        "rotated": [-torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17],
    }

    URDF_QPOS_MAP = {
        "middle": [0, 1.75, -1.5, -0.6, 0, -0.15],
        "zero": [0, 3.3, -3.1, -0.54, 0, 0],
        "rotated": [-1.6, 1.79, -1.6, 0.95, -1.7, 1.63],
        "rest": [0, 0, 0, 0.7, 0, 0],
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

    GOLF_BALL_RADIUS = 4.27e-2 / 2

    INCHES_TO_METERS = 0.0254

    DISTANCE_THRESHOLD = 0.01 # the threshold for the distance between the golf ball and the cup for the ball to be considered in the cup, or for the ball to be considered moved from the initial position

    DIFF_THRESHOLD = 200 # the threshold for the difference between the leader and follower gripper positions for the hold state to be detected

    def logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False # avoid propagating to the root logger, otherwise each log line shows twice in stderr
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger