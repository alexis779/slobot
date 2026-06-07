from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig


@TeleoperatorConfig.register_subclass("slobot_so100_leader")
@dataclass
class SlobotSO100LeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    show_gui: bool = False  # Tk on main thread; pumped from get_teleop_events (required for online RL)
    default_intervention: bool = False  # True = human teleop on start; False = policy until 'i' is pressed
