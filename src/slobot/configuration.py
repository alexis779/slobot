import torch
import logging
import os

class Configuration:
    WORK_DIR = f"{os.environ['HOME']}/.slobot"

    # 4:3 aspect ratio
    QVGA = (320, 240)
    VGA = (640, 480)
    XGA = (1024, 768)
    UXGA = (1600, 1200)

    QPOS_MAP = {
        "middle": [0, -torch.pi/2, torch.pi/2, 0, 0, -0.15],
        "middle_interpolated": [0.0000, -1.5730,  1.4830,  0.0000,  0.0000,  0.7880],
        "zero": [0, 0, 0, 0, 0, 0],
        "zero_interpolated": [0, 0, 0, 0, 0, 0],
        "zero_min_error": [0, 0, 0, 0, 0, -0.1748],
        "rotated": [-torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17],
        "episode01": [0, 0, 0, 0, 0, 0],
        "episode02": [0, 0, 0, 0, 0, 0],
        "episode05": [0.9241, -1.5767,  2.0920,  1.0622, -2.4949, -0.1748],
    }

    POS_MAP = {
        "middle": [2047, 2047, 2047, 2047, 2047, 2047],
        "middle_interpolated": [2082, 1998, 1999, 2016, 2047, 2614],
        "zero": [2047, 3083, 1030, 2048, 2047, 2144],
        "zero_interpolated": [2061, 2936, 1024, 2078, 2020, 2049],
        "zero_min_error": [2063, 3033, 994, 2092, 333, 1902],
        "rotated": [3071, 2052, 2051, 3071, 1023, 3168],
        "rest": [2016, 907, 3070, 2831, 1937, 2035],
        "episode01": [2075, 2929, 994, 2014, 2043, 2046],
        "episode02": [2054, 3039, 929, 2168, 1980, 2054],
        "episode05": [1465, 1970, 2448, 2742, 423, 1902],
    }

    REFERENCE_FRAME = 'episode01'

    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    GRIPPER_ID = 5 # the id of the jaw joint

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