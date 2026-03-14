import os

from slobot.lerobot.episode_replayer import EpisodeReplayer
from slobot.simulation_frame import SimulationFrame

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class SimDatasetGenerator:
    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]
        self.sim_repo_id = kwargs["sim_repo_id"]

        kwargs["show_viewer"] = False
        kwargs["n_envs"] = 1

        kwargs["rgb"] = True
        kwargs["step_handler"] = self

        # Use episode 0 for init to get meta without loading full dataset
        init_episode_ids = kwargs.get("episode_ids") or [0]
        kwargs["episode_ids"] = init_episode_ids
        self.episode_replayer = EpisodeReplayer(**kwargs)
        self.ds_meta = self.episode_replayer.episode_loader.dataset.meta
        self.task = self.ds_meta.tasks.index[0]

    def generate_dataset(self, episode_ids=None):
        # Use source dataset features - same structure, sim images replace real
        source_features = dict(self.ds_meta.features)
        dataset_features = source_features

        self.dataset = LeRobotDataset.create(
            repo_id=self.sim_repo_id,
            fps=self.ds_meta.fps,
            features=dataset_features,
            robot_type=self.ds_meta.robot_type,
            use_videos=True,
            image_writer_threads=4,
        )

        if episode_ids is None:
            episode_ids = range(self.ds_meta.total_episodes)

        for episode_id in episode_ids:
            self.generate_episode(episode_id)

        self.dataset.finalize()
        self._push_to_hub()

    def generate_episode(self, episode_id):
        episode_dataset = self.episode_replayer.load_dataset(episode_id)

        self.episode_replayer.load_episodes([episode_id])
        self.episode_replayer.set_object_initial_positions()

        for frame_id, row in enumerate(episode_dataset):
            self.episode_replayer.replay_frame(frame_id)
            self.generate_frame(row)

        self.dataset.save_episode()

    def generate_frame(self, row):
        import numpy as np

        frame = {}

        # Action and observation.state - convert to numpy if needed
        action = row["action"]
        obs_state = row["observation.state"]
        if hasattr(action, "numpy"):
            action = action.numpy()
        if hasattr(obs_state, "numpy"):
            obs_state = obs_state.numpy()
        if hasattr(action, "tolist"):
            action = np.array(action) if not isinstance(action, np.ndarray) else action
        if hasattr(obs_state, "tolist"):
            obs_state = np.array(obs_state) if not isinstance(obs_state, np.ndarray) else obs_state

        frame["action"] = action
        frame["observation.state"] = obs_state
        # Map camera keys: first = fixed side camera, second = mobile link camera
        for i, cam_key in enumerate(self.ds_meta.camera_keys):
            frame[cam_key] = self.current_images[i] if i < len(self.current_images) else self.current_images[0]
        frame["task"] = self.task

        self.dataset.add_frame(frame)

    def handle_step(self, simulation_frame: SimulationFrame):
        # Fixed side camera (index 0), mobile link camera (index 1)
        self.current_images = [
            simulation_frame.side_camera_frame.rgb,
            simulation_frame.link_camera_frame.rgb,
        ]

    def _push_to_hub(self):
        """Upload the dataset to Hugging Face Hub. Uses HF_TOKEN from env if set."""
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(repo_id=self.sim_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=self.sim_repo_id,
            folder_path=str(self.dataset.root),
            repo_type="dataset",
            ignore_patterns=["images/"],
            token=token,
        )
