from slobot.lerobot.episode_replayer import EpisodeReplayer
from slobot.simulation_frame import SimulationFrame
from slobot.feetech import Feetech

from lerobot.datasets.v2.convert_dataset_v1_to_v2 import make_robot_config
from lerobot.robots import make_robot_from_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

class SimDatasetGenerator:
    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]
        self.sim_repo_id = kwargs["sim_repo_id"]

        kwargs["show_viewer"] = False
        kwargs["n_envs"] = 1

        kwargs["rgb"] = True
        kwargs["step_handler"] = self

        self.episode_replayer = EpisodeReplayer(**kwargs)
        self.task = self.episode_replayer.ds_meta.tasks[0]

    def generate_dataset(self, episode_ids=None):
        camera_config = {"sim": OpenCVCameraConfig(index_or_path=self.video_filename, width=self.episode_replayer.res[0], height=self.episode_replayer.res[1], fps=self.episode_replayer.ds_meta.fps)}
        robot_config = make_robot_config(Feetech.ROBOT_TYPE, port=Feetech.PORT0, id=Feetech.FOLLOWER_ID, cameras=camera_config)
        self.robot = make_robot_from_config(robot_config)

        self.cam_key = next(self.robot.cameras.keys())

        action_features = hw_to_dataset_features(self.robot.action_features, "action")
        obs_features = hw_to_dataset_features(self.robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        self.dataset = LeRobotDataset.create(
            repo_id=self.sim_repo_id,
            fps=self.episode_replayer.ds_meta.fps,
            features=dataset_features,
            robot_type=self.robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

        if episode_ids is None:
            episode_ids = range(self.episode_replayer.ds_meta.total_episodes)

        for episode_id in episode_ids:
            self.generate_episode(episode_id)

        self.dataset.push_to_hub()

    def generate_episode(self, episode_id):
        episode_dataset = self.episode_replayer.load_dataset(episode_id)

        self.episode_replayer.load_episodes([episode_id])
        self.episode_replayer.set_object_initial_positions()

        for frame_id, row in enumerate(episode_dataset):
            self.episode_replayer.replay_frame(frame_id)
            self.generate_frame(row)
    
        self.dataset.save_episode()

    def generate_frame(self, row):
        action = {}
        observation_state = {}

        for motor_id, motor in self.robot.bus.motors.items():
            motor_key = f"{motor_id}.pos"
            action[motor_key] = row['action'][motor.id-1]
            observation_state[motor_key] = row['observation.state'][motor.id-1]

        action = build_dataset_frame(self.dataset.features, action, prefix="action")

        observation_state[self.cam_key] = self.current_image
        observation_state = build_dataset_frame(self.dataset.features, observation_state, prefix="observation")

        frame = {**action, **observation_state}

        self.dataset.add_frame(frame, self.task)

    def handle_step(self, simulation_frame: SimulationFrame):
        image = simulation_frame.rgb
        self.current_image = image
