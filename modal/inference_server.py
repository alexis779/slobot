from dataclasses import dataclass

from huggingface_hub import snapshot_download

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import So100DataConfig
from gr00t.model.policy import Gr00tPolicy

class CustomSo100DataConfig(So100DataConfig):
    video_keys = ["video.phone"]

# Create a policy
# The `Gr00tPolicy` class is being used to create a policy object that encapsulates
# the model path, transform name, embodiment tag, and denoising steps for the robot
# inference system. This policy object is then utilized in the server mode to start
# the Robot Inference Server for making predictions based on the specified model and
# configuration.

# we will use an existing data config to create the modality config and transform
# if a new data config is specified, this expect user to
# construct your own modality config and transform
# see gr00t/utils/data.py for more details

@dataclass
class ServerArgs:
    #data_config: str = "so100"
    repo_id: str = "alexis779/so100_ball_cup_gr00t"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: int = 4
    port: int = 7860

args = ServerArgs()

def robot_inference_server(port: int):
    model_path = args.repo_id.split("/")[1]
    snapshot_download(repo_id=args.repo_id, local_dir=model_path)

    data_config = CustomSo100DataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    return RobotInferenceServer(policy, port=port)

### app

import modal

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image
    .from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install("huggingface_hub")
    .pip_install("git+https://github.com/NVIDIA/Isaac-GR00T.git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
    )
    .pip_install(  # add flash-attn
        "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    ))

app = modal.App("gr00t-inference-server")

@app.function(image=image, gpu="any", timeout=3600)
#@modal.web_server()
def inference_server():
    with modal.forward(args.port, unencrypted=True) as tunnel:
        # keep track of unencrypted_host and unencrypted_port in the generated tunnel info below
        print(f"tunnel = {tunnel}")
        server = robot_inference_server(args.port)
        server.run()