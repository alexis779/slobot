from dataclasses import dataclass

from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("slobot_so100_follower")
@dataclass
class SlobotSO100FollowerConfig(RobotConfig, SOFollowerConfig):
    pass
