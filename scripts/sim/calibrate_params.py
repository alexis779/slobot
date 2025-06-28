import modal

from slobot.configuration import Configuration
from slobot.lerobot.episode_replayer import EpisodeReplayer

image = (
    modal.Image
    .from_registry("debian:trixie-slim", add_python="3.12")
    .apt_install(
        "git",
        "clang",
        "libglib2.0-0",
        "libxrender1",
        "libgl1",
        "libegl1",
    )
    .run_commands(
        "pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git",
        "pip install https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "pip install git+https://github.com/huggingface/lerobot.git",
        "pip install slobot==0.1.14",
    )
)

app = modal.App("slobot")

@app.function(image=image, gpu="any")
def replay_episode(dataset_repo_id: str, episode_ids: str):
    mjcf_path = Configuration.MJCF_CONFIG
    episode_replayer = EpisodeReplayer(repo_id=dataset_repo_id, mjcf_path=mjcf_path, show_viewer=False)

    episode_ids = [int(epipsode_id.strip()) for epipsode_id in episode_ids.split(',')]
    for episode_id in episode_ids:
        episode_replayer.replay_episode(episode_id)

