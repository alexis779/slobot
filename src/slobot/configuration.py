import torch
import logging

class Configuration:
    WORK_DIR = "/tmp/slobot"

    # 4:3 aspect ratio
    QVGA = (320, 240)
    VGA = (640, 480)
    XGA = (1024, 768)
    UXGA = (1600, 1200)

    QPOS_MAP = {
        "middle": [0, -torch.pi/2, torch.pi/2, 0, 0, -0.15],
        "zero": [0, 0, 0, 0, 0, 0],
        "zero_interpolated": [0, 0, 0, 0, 0, 0],
        "rotated": [-torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2],
        "rest": [0.049, -3.32, 3.14, 1.21, -0.17, -0.17],
        "episode02": [0.5431, -0.5907,  0.5669,  1.6152, -2.1138, -0.1748],
        "episode05": [0.9241, -1.5767,  2.0920,  1.0622, -2.4949, -0.1748],
    }

    POS_MAP = {
        "middle": [2047, 2047, 2047, 2047, 2047, 2047],
        "zero": [2047, 3083, 1030, 2048, 2047, 2144],
        "zero_interpolated": [2055, 2939, 1031, 2078, 2020, 2049],
        "rotated": [3071, 2052, 2051, 3071, 1023, 3168],
        "rest": [2016, 907, 3070, 2831, 1937, 2035],
        "episode02": [1700, 2663, 1317, 3158, 590, 1902],
        "episode05": [1465, 1970, 2448, 2742, 423, 1902],
    }

    REFERENCE_FRAME = 'zero_interpolated'

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