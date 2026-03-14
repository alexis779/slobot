"""Convert rerun.io recordings to LeRobot dataset format."""

import gc
import io
import logging
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import rerun as rr
import torch

from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics

LOG = logging.getLogger(__name__)

# Rerun entity paths for sim recordings
_SIM_QPOS_PREFIX = "/sim/qpos"
_LEADER_QPOS_PREFIX = "/leader/qpos"
_FOLLOWER_QPOS_PREFIX = "/follower/qpos"

# Video paths in rerun (maps to LeRobot camera keys)
# Try sim paths first, then webcam paths for teleop recordings
VIDEO_PATHS = {
    "observation.images.phone": [
        "/sim/side/video:VideoStream:sample",
        "/webcam2/video:VideoStream:sample",
    ],
    "observation.images.gripper": [
        "/sim/link/video:VideoStream:sample",
        "/webcam4/video:VideoStream:sample",
    ],
}

DEFAULT_TASK = "Grab the golf ball and put it in the cup"


def _get_scalar_column(table: pa.Table, metric_prefix: str) -> torch.Tensor | None:
    """Extract scalar tensor from arrow table for a metric (e.g. /sim/qpos)."""
    columns = []
    for joint_name in Configuration.JOINT_NAMES:
        col_name = f"{metric_prefix}/{joint_name}:Scalars:scalars"
        if col_name not in table.column_names:
            return None
        column = table.column(col_name)
        values = []
        for i in range(len(table)):
            cell = column[i]
            if hasattr(cell, "as_py"):
                cell = cell.as_py()
            if cell is None or (isinstance(cell, list) and len(cell) == 0):
                values.append(float("nan"))
            else:
                val = cell[0] if isinstance(cell, list) else cell
                values.append(float(val))
        columns.append(values)

    if not columns:
        return None

    tensor = torch.tensor(columns, dtype=torch.float32).T
    mask = torch.isnan(tensor)
    if mask.any():
        idx = torch.where(~mask, torch.arange(len(tensor), device=tensor.device), 0)
        idx = torch.cummax(idx, dim=0).values
        tensor = tensor[idx]
    return tensor


def _decode_video_samples(table: pa.Table, col_name: str) -> list[np.ndarray] | None:
    """Decode video samples from rerun VideoStream column to RGB frames."""
    if col_name not in table.column_names:
        return None

    column = table.column(col_name)
    samples = []
    for i in range(len(table)):
        val = column[i]
        if val is None:
            continue
        if hasattr(val, "as_py"):
            val = val.as_py()
        if val is None or (isinstance(val, list) and len(val) == 0):
            continue
        if isinstance(val[0], list):
            sample = bytes(val[0])
        else:
            sample = bytes(val)
        samples.append(sample)

    if not samples:
        return None

    try:
        all_data = b"".join(samples)
        container = av.open(io.BytesIO(all_data), format="h264")
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        return frames
    except Exception:
        return None


def _load_rerun_episode(rrd_file: str) -> dict | None:
    """
    Load a single rerun recording as episode data for LeRobot.

    Returns dict with keys: action, observation.state, and camera keys.
    Returns None if the recording is empty or has no usable data.
    """
    with rr.server.Server(datasets={RerunMetrics.APPLICATION_ID: [rrd_file]}) as server:
        client = server.client()
        dataset = client.get_dataset(RerunMetrics.APPLICATION_ID)
        df = dataset.reader(index=RerunMetrics.TIME_METRIC, fill_latest_at=True)
        table = df.to_arrow_table()

    if table.num_rows == 0:
        return None

    obs_state = _get_scalar_column(table, _FOLLOWER_QPOS_PREFIX)
    if obs_state is None:
        obs_state = _get_scalar_column(table, _SIM_QPOS_PREFIX)
    if obs_state is None:
        return None

    action = _get_scalar_column(table, _LEADER_QPOS_PREFIX)
    if action is None:
        action = torch.roll(obs_state, -1, dims=0)
        action[-1] = obs_state[-1]

    n_frames = len(obs_state)
    if n_frames == 0:
        return None

    images = {}
    for cam_key, col_names in VIDEO_PATHS.items():
        col_names = col_names if isinstance(col_names, list) else [col_names]
        frames = None
        for col_name in col_names:
            frames = _decode_video_samples(table, col_name)
            if frames is not None and len(frames) > 0:
                break
        if frames is not None and len(frames) >= n_frames:
            images[cam_key] = frames[:n_frames]
        elif frames is not None and len(frames) > 0:
            if len(frames) < n_frames:
                last_frame = frames[-1]
                frames = frames + [last_frame] * (n_frames - len(frames))
            images[cam_key] = frames[:n_frames]

    if not images:
        return None

    first_cam = next(iter(images.values()))
    frame_shape = first_cam[0].shape

    for cam_key in VIDEO_PATHS:
        if cam_key not in images:
            images[cam_key] = [
                np.zeros(frame_shape, dtype=np.uint8) for _ in range(n_frames)
            ]

    return {
        "action": action.numpy().astype(np.float32),
        "observation.state": obs_state.numpy().astype(np.float32),
        "images": images,
        "n_frames": n_frames,
        "frame_shape": frame_shape,
    }


def _get_dataset_features(
    fps: int,
    image_height: int,
    image_width: int,
    camera_keys: tuple[str, ...],
) -> dict:
    """Return LeRobot dataset features for so100 sim recordings."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
            "fps": fps,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
            "fps": fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for cam_key in camera_keys:
        features[cam_key] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": image_height,
                "video.width": image_width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features


class RerunToLeRobotConverter:
    """Convert rerun.io recordings to a LeRobot dataset.

    Each .rrd file becomes one episode. Empty recordings are skipped.
    Episodes are processed one at a time to limit memory usage.
    """

    def __init__(
        self,
        rerun_dir: Path | str,
        dataset_id: str,
        *,
        episode_ids: list[int] | None = None,
        task: str = DEFAULT_TASK,
    ):
        self.rerun_dir = Path(rerun_dir)
        self.dataset_id = dataset_id
        self.episode_ids = episode_ids
        self.task = task

    def convert(self) -> int:
        """Run the conversion. Returns the number of episodes converted."""
        rrd_files = self._resolve_rrd_files()
        LOG.info("Converting episodes to LeRobot dataset (processing one at a time to limit memory)")
        first_ep, first_rrd_name = self._load_first_episode(rrd_files)

        features = self._get_features_from_episode(first_ep)
        dataset = self._create_dataset(features)
        camera_keys = tuple(VIDEO_PATHS.keys())

        episode_count = 0
        LOG.info("Adding episode 0 from %s (%d frames)", first_rrd_name, first_ep["n_frames"])
        self._add_episode_to_dataset(first_ep, dataset, camera_keys)
        episode_count = 1
        del first_ep
        gc.collect()

        for rrd_path in rrd_files[1:]:
            episode = _load_rerun_episode(str(rrd_path))
            if episode is not None:
                LOG.info(
                    "Adding episode %d from %s (%d frames)",
                    episode_count,
                    rrd_path.name,
                    episode["n_frames"],
                )
                self._add_episode_to_dataset(episode, dataset, camera_keys)
                episode_count += 1
                del episode
                gc.collect()
            else:
                LOG.info("Skipping empty recording: %s", rrd_path.name)

        dataset.finalize()
        LOG.info("Dataset saved. Total episodes: %d", episode_count)
        return episode_count

    def _resolve_rrd_files(self) -> list[Path]:
        """Resolve and filter the list of .rrd files to process."""
        if not self.rerun_dir.is_dir():
            raise FileNotFoundError(f"Rerun directory not found: {self.rerun_dir}")

        rrd_files = sorted(self.rerun_dir.glob("*.rrd"))
        if not rrd_files:
            raise FileNotFoundError(f"No .rrd files found in {self.rerun_dir}")

        if self.episode_ids is not None:
            rrd_files = [rrd_files[i] for i in self.episode_ids if i < len(rrd_files)]
            if not rrd_files:
                raise ValueError(f"No valid episode IDs in {self.episode_ids}")
            LOG.info("Processing %d episodes (ids %s)", len(rrd_files), self.episode_ids)
        else:
            LOG.info("Found %d .rrd files in %s", len(rrd_files), self.rerun_dir)

        return rrd_files

    def _load_first_episode(self, rrd_files: list[Path]) -> tuple[dict, str]:
        """Load the first non-empty episode. Returns (episode_data, rrd_name)."""
        for rrd_path in rrd_files:
            episode = _load_rerun_episode(str(rrd_path))
            if episode is not None:
                return episode, rrd_path.name
            LOG.info("Skipping empty recording: %s", rrd_path.name)

        raise ValueError("No non-empty recordings found. All recordings were skipped.")

    def _get_features_from_episode(self, episode: dict) -> dict:
        """Build dataset features dict from episode frame shape."""
        h, w, c = episode["frame_shape"]
        camera_keys = tuple(VIDEO_PATHS.keys())
        return _get_dataset_features(
            fps=30,
            image_height=h,
            image_width=w,
            camera_keys=camera_keys,
        )

    def _create_dataset(self, features: dict):
        """Create the LeRobot dataset."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset.create(
            repo_id=self.dataset_id,
            fps=30,
            features=features,
            robot_type="so100_follower",
            use_videos=True,
            image_writer_threads=4,
        )

    def _add_episode_to_dataset(
        self,
        episode: dict,
        dataset,
        camera_keys: tuple[str, ...],
    ) -> None:
        """Add a single episode's frames to the dataset."""
        for frame_idx in range(episode["n_frames"]):
            action = np.asarray(episode["action"][frame_idx], dtype=np.float32).flatten()
            obs_state = np.asarray(
                episode["observation.state"][frame_idx], dtype=np.float32
            ).flatten()
            frame = {
                "action": action,
                "observation.state": obs_state,
                "task": self.task,
            }
            for cam_key in camera_keys:
                frame[cam_key] = episode["images"][cam_key][frame_idx]
            dataset.add_frame(frame)
        dataset.save_episode()


# Backwards compatibility: expose load_rerun_episode for callers that use it directly
def load_rerun_episode(rrd_file: str) -> dict | None:
    """Load a single rerun recording as episode data. See RerunToLeRobotConverter."""
    return _load_rerun_episode(rrd_file)


def get_default_dataset_features(
    fps: int = 30,
    image_height: int = 480,
    image_width: int = 640,
    camera_keys: tuple[str, ...] = ("observation.images.phone", "observation.images.gripper"),
) -> dict:
    """Return default LeRobot dataset features. See RerunToLeRobotConverter."""
    return _get_dataset_features(fps, image_height, image_width, camera_keys)
