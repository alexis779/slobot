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
        "pip install https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "pip install git+https://github.com/huggingface/lerobot.git",
        "pip install slobot==0.1.14",
        "mkdir -p /run/sshd",
    ).env(
         {"LD_LIBRARY_PATH": "/lib/x86_64-linux-gnu"} # fixes a dynamic library resolution error when importing torchcodec after genesis
    ).add_local_file(
        f"{os.environ['HOME']}/.ssh/id_ed25519.pub", "/root/.ssh/authorized_keys", # enable ssh connection to modal container
        copy=True
    ).add_local_file(
        f"{os.environ['HOME']}/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_arm.json", "/root/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_arm.json",
        copy=True
    ).add_local_python_source(
        "lerobot",
        "slobot",
        ignore=~modal.FilePatternMatcher("**/*.py", "**/config/*")
    )
)

app = modal.App("slobot")

@app.function(image=image, timeout=3600)
def open_ssh_tunnel():
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    with modal.forward(port=22, unencrypted=True) as tunnel:
        hostname, port = tunnel.tcp_socket
        connection_cmd = f'ssh -p {port} root@{hostname}'
        print(f"ssh into container using: {connection_cmd}")
        time.sleep(3600)  # keep alive for 1 hour or until killed

@app.function(image=image, gpu="any", timeout=3600)
def replay_episodes(dataset_repo_id: str, episode_ids: str = None):
    mjcf_path = Configuration.MJCF_CONFIG
    episode_replayer = EpisodeReplayer(repo_id=dataset_repo_id, mjcf_path=mjcf_path, show_viewer=False)

    if not episode_ids:
         episode_replayer.replay_episodes()
    else:
        episode_ids = [int(epipsode_id.strip()) for epipsode_id in episode_ids.split(',')]
        for episode_id in episode_ids:
            episode_replayer.replay_episode(episode_id)

