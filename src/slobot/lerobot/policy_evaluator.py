import time

import torch
import cv2
import numpy as np

from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import make_robot_config
from lerobot.common.policies.factory import get_policy_class
from lerobot.common.robots import make_robot_from_config
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import auto_select_torch_device


class PolicyEvaluator:
    INFERENCE_TIME_S = 60
    FPS = 25

    def __init__(self, robot_type, policy_type, model_path, port):
        self.model_path = model_path
        self.policy_type = policy_type

        robot_config = make_robot_config(robot_type, port=port)
        self.robot = make_robot_from_config(robot_config)

        self.device = auto_select_torch_device()

    def evaluate(self):
        self.robot.connect()

        policy_cls = get_policy_class(self.policy_type)

        self.policy = policy_cls.from_pretrained(self.model_path)
        self.policy.to(self.device)

        for step in range(self.INFERENCE_TIME_S * self.FPS):
            self.run_step(step)

    def run_step(self, step):
        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = self.robot.capture_observation()

        image = observation['observation.images.phone']
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"eval_images/img_{step:04d}.jpg", np_image)

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(self.device)

        # Compute the next action with the policy
        # based on the current observation
        action = self.policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        self.robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / self.FPS - dt_s)