import modal
import subprocess
import time
import os

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
        "ffmpeg",
        "openssh-server",
    )
    .run_commands(
        "pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git",
        "pip install git+https://github.com/huggingface/lerobot.git",
        "pip install slobot==0.1.16",
        "mkdir -p /run/sshd",
    ).env(
         {"LD_LIBRARY_PATH": "/lib/x86_64-linux-gnu"} # fixes a dynamic library resolution error when importing torchcodec after genesis
    ).add_local_file(
        f"{os.environ['HOME']}/.ssh/id_ed25519.pub", "/root/.ssh/authorized_keys", # enable ssh connection to modal container
        copy=True
    ).add_local_python_source(
        "slobot",
        ignore=~modal.FilePatternMatcher("**/*.py", "**/config/*")
    )
)

app = modal.App("slobot")
vol = modal.Volume.from_name("huggingface_cache", create_if_missing=True)
cache_folder = "/root/.cache/huggingface"

# modal volume put huggingface_cache ~/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_arm.json lerobot/calibration/robots/so100_follower/follower_arm.json

@app.function(image=image, timeout=3600)
def open_ssh_tunnel():
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    with modal.forward(port=22, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f'ssh -p {port} root@{hostname}'
        print(f"ssh into container using: {connection_cmd}")
        time.sleep(3600)  # keep alive for 1 hour or until killed

@app.function(image=image, gpu="any", timeout=3600, volumes={cache_folder: vol})
def replay_episodes(dataset_repo_id: str, episode_ids: str = None):
    mjcf_path = Configuration.MJCF_CONFIG

    episode_count = None
    episode_ids = None
    if episode_ids:
        episode_ids = [int(epipsode_id.strip()) for epipsode_id in episode_ids.split(',')]
        episode_count = len(episode_ids)

    episode_replayer = EpisodeReplayer(repo_id=dataset_repo_id, mjcf_path=mjcf_path, show_viewer=False, n_envs=episode_count)

    episode_replayer.load_episodes(episode_ids=episode_ids)
    rate = episode_replayer.replay_episodes()
    print(f"Success rate: {rate:.2f} for episode_ids = {episode_ids}")
