from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

mjcf_path = Configuration.MJCF_CONFIG
repo_id = "alexis779/so100_ball_cup"
episode_replayer = EpisodeReplayer(repo_id=repo_id, mjcf_path=mjcf_path)
episode_id = 1
episode_replayer.replay_episode(episode_id)
