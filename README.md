# LeRobot SO-ARM-100 6 DOF robotic arm manipulation with Genesis simulator and Feetech motors

There are 2 main use cases
1. sim to real, where genesis controls the physical robot
2. real to sim, where the physical robot moves will refresh the robot rendering in genesis

## Acknowledgements

### The Robot Studio

[SO-ARM-100](https://github.com/TheRobotStudio/SO-ARM100) provides CAD & [STL model](https://github.com/TheRobotStudio/SO-ARM100/blob/main/stl_files_for_3dprinting/Follower/Print_Follower_SO_ARM100_08k_Ender.STL) of the robotic arm links. After 3D-printing them and ordering the remaining parts, the robot can be assembled for prototyping.

### Feetech

[Feetech STS3115](https://www.feetechrc.com/74v-19-kgcm-plastic-case-metal-tooth-magnetic-code-double-axis-ttl-series-steering-gear.html) is the servo inserted in each joint of the robot.

- It's *motor* rotates the connected link.
- It's *magnetic encoder* measures the absolute angular position of the joint.

### LeRobot

[LeRobot](https://github.com/huggingface/lerobot) provides SOTA policies to perform *Imitation Learning* and *Reinforcement Learning* in robotics.

Tele-operate to create a new dataset: have the follower arm perform a task of interest, by replicating the motion of the leader arm held a human operator.

Then fine-tune a model on the training dataset.

Finally evaluate it on the eval dataset to see how well it performs.

```
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

### Genesis

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) is the physics engine running the simulation.

```
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```

## Topics

- [Tele-operation](doc/teleoperate.md)

## Policies

### ACT

Action Chunking Transformer is part of the [Aloha paper](https://tonyzhaozh.github.io/aloha/).

### PI0

PI0 was released by Physical Intelligence to provide an OpenSource policy similar to Figure AI Helix architecture.

### Isaac Gr00t N1

[NVIDIA Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T) was released by Nvidia.

### SmolVLA

SmolVLA was released by HuggingFace, using LeRobot datasets for training.

## Setup the environment

### Python

Python version should match [OMPL library](https://github.com/ompl/ompl/releases/tag/prerelease) compatible version.

Following installs Python `3.12.9` with *pyenv*


```
sudo apt install pyenv
python_version=3.12.9
pyenv install $python_version
export PATH="$HOME/.pyenv/versions/$python_version/bin:$PATH"
```

Create a virtual environment with *venv*

```
python -m venv .venv
. .venv/bin/activate
```

Install following dependencies


### 1. slobot

```
git clone git+https://github.com/alexis779/slobot.git
cd slobot
pip install -e .
```

### 2. Robot Configuration

Ensure the robot [configuration](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) in available in `slobot.config` package.

```
cd ..
git clone https://github.com/google-deepmind/mujoco_menagerie
cd slobot
ln -s ../mujoco_menagerie/trs_so_arm100 src/slobot/config/trs_so_arm100
```

### 3. LeRobot

```
pip install git+https://github.com/huggingface/lerobot.git
```

### 4. Genesis

```
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

Also refer to the [installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html). Make sure to run the [hello world example](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html) successfully.

##### Known issue

On Ubuntu, Qt5 library may be incompatible with [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) native library. See [reported issue](https://github.com/Genesis-Embodied-AI/Genesis/issues/189). As a workaround, give precedence to the *python module* QT library instead of the *Ubuntu system* QT library.

```
SITE_PACKAGES=`pip show pymeshlab | grep Location | sed 's|Location: ||'`
PYMESHLAB_LIB=$SITE_PACKAGES/pymeshlab/lib
```

Make sure the symbol is found

```
strings $PYMESHLAB_LIB/libQt5Core.so.5 | grep _ZdlPvm
```

Finally, configure `LD_LIBRARY_PATH` to overwrite QT library path,

```
LD_LIBRARY_PATH=$PYMESHLAB_LIB PYOPENGL_PLATFORM=glx python <script.py>
```

### 5. OMPL

```
pip install https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl
```


## Validation & Calibration

A series of scripts are provided to help with validation and calibration.

### 0. Validate the preset qpos in sim

This validates that the robot is in the targetted position preset from the sim qpos.

```
PYOPENGL_PLATFORM=glx python scripts/validation/0_validate_sim_qpos.py [middle|zero|rotated|rest]
```

| middle | zero | rotated | rest |
|-|-|-|-|
| ![middle](doc/SimMiddle.png) | ![zero](doc/SimZero.png) | ![rotated](doc/SimRotated.png) | ![rotated](doc/SimRest.png) |


### 1. Validate the preset pos

For [motor calibration](https://huggingface.co/docs/lerobot/so101#calibration-video), LeRobot suggests the `middle` position, where all the joints are positioned in the middle of their range.

Position the arm manually into the `middle` preset.

```
python scripts/validation/1_calibrate_motor_pos.py middle
```

It will read the motor positions and output them. It should return an int vector around `[2047, 2047, 2047, 2047, 2047, 2047]`, the middle position for each motor.

### 2. Validate the preset *pos to qpos* conversion in sim

Same as script 0, but using the motor pos instead of the sim qpos.

```
PYOPENGL_PLATFORM=glx python scripts/validation/2_validate_sim_pos.py [middle|zero|rotated|rest]
```

### 3. Validate the preset pos in real

Similar than 2 which is in sim but now in real. It validates the robot is positioned correctly to the target pos.

```
python scripts/validation/3_validate_real_pos.py [middle|zero|rotated|rest]
```

### 4. Validate real to sim

This validates that moving the real robot also updates the rendered robot in sim.

```
PYOPENGL_PLATFORM=glx python scripts/validation/4_validate_real_to_sim.py [middle|zero|rotated|rest]
```

### 5. Validate sim to real

This validates the robot simulation also controls the physical robot.

```
PYOPENGL_PLATFORM=glx python scripts/validation/5_validate_sim_to_real.py [middle|zero|rotated|rest]
```


## Examples

### Real

This example moves the robot to the preset positions, waiting 1 sec in between each one.

```
python scripts/real.py
```

<video controls src="https://github.com/user-attachments/assets/857dd958-2e4c-4221-abef-563f9617385a"></video>


### Sim To Real

This example performs the 3 elemental rotations in sim and real.
The simulation generates steps, propagating the joint positions to the Feetech motors.

```
PYOPENGL_PLATFORM=glx python scripts/sim_to_real.py
```


| sim | real |
|----------|-------------|
| <video controls src="https://github.com/user-attachments/assets/eab20130-a21d-4811-bca8-07502012b8da"></video> | <video controls src="https://github.com/user-attachments/assets/a429d559-58e4-4328-a7f0-17f7477125ff"></video> |


### Real To Sim

Consider a [LeRobot dataset](https://huggingface.co/datasets/alexis779/so100_ball_cup) created via teleoperation.

*Visualize Dataset* [HuggingFace space](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Falexis779%2Fso100_ball_cup%2Fepisode_1%3Ft%3D23) is an interactive tool to visualize episodes from the dataset.

Following script replays the tele-operated episode robot state `qpos` from each frame into the simulation environment, using the same FPS as the camera.

```
PYOPENGL_PLATFORM=glx python scripts/sim/replay_episodes.py --dataset_repo_id alexis779/so100_ball_cup2 --episode_id 0
```

| real | sim |
|----------|-------------|
| <video controls src="https://github.com/user-attachments/assets/fbaef5c9-7767-4aff-b879-4f7f59e993c4"></video> | <video controls src="https://github.com/user-attachments/assets/f260e84e-946b-4c74-9e06-4e36dab704a2"></video> |


### Image stream

Genesis camera provides access to each frames rendered by the rasterizer. Multiple types of image are provided:
- RGB
- Depth
- Segmentation
- Surface

The following script iterates through all the frames, calculating the FPS metric every second.

```
PYOPENGL_PLATFORM=glx python scripts/sim_fps.py
...
FPS= FpsMetric(1743573645.3103304, 0.10412893176772242)
FPS= FpsMetric(1743573646.3160942, 59.656155690238116)
FPS= FpsMetric(1743573647.321373, 59.68493363485116)
FPS= FpsMetric(1743573649.8052156, 12.078059963768446)
FPS= FpsMetric(1743573650.8105915, 59.67917299445178)
FPS= FpsMetric(1743573651.8152244, 59.723304924655935)
...
```


### Gradio apps

Gradio app is a UI web framework to demo ML applications.

Navigate to the [local URL](http://127.0.0.1:7860) in the browser. Then click *Run* button.

#### Image

The [`Image` component](https://www.gradio.app/docs/gradio/image) can sample the frames of the simulation at a small FPS rate.
The frontend receives backend events via a Server Side Event stream. For each new *frame generated* event, it downloads the image from the webserver and displays it to the user.

```
PYOPENGL_PLATFORM=egl python scripts/sim_gradio_image.py
```

![Genesis frame types](./doc/GenesisImageFrameTypes.png)

#### Video

The [`Video` component](https://www.gradio.app/docs/gradio/video) can play a full mp4 encoded in h264 or a stream of smaller TS files.

```
PYOPENGL_PLATFORM=egl python scripts/sim_gradio_video.py
```

![Genesis frame types](./doc/GenesisVideoFrameTypes.png)


#### Qpos

The qpos app displays the joint angular position numbers.

![Genesis qpos](./doc/GenesisQpos.png)

```
python scripts/sim_gradio_qpos.py

2025-04-02 00:45:17,551 - INFO - Sending qpos [1.4888898134231567, -1.8273500204086304, 2.3961710929870605, -0.5487295389175415, 1.5706498622894287, -2.59892603935441e-05]
```

A client connects to the server to receive the qpos updates.

It can then dispatch them to the robot at a predefined `fps` rate to control its position.

```
python scripts/sim_to_real_client.py

2025-04-02 00:45:17,764 - INFO - Received qpos (1.49, -1.83, 2.4, -0.55, 1.57, -0.0)
```

#### Plot

The [Plot component](https://www.gradio.app/docs/gradio/plot) can display a chart. Dashboard monitors key metrics in dedicated [Tab](https://www.gradio.app/docs/gradio/tab)s.
- **qpos**, in *rad*
- **velocity**, in *rad/sec*
- **control force**, in *N.m*


```
python scripts/sim_gradio_dashboard.py
```

![Gradio dashboard](./doc/GradioTabPlots.png)

# Camera feed

Install **Webcam IP** Android app on your phone, select 640 x 480 image resolution and start server.

List v4l2 devices

```
v4l2-ctl --list-devices
```

Create a looback device if /dev/video4 is missing in the above output.

```
sudo modprobe v4l2loopback devices=1 video_nr=4
```

Create a virtual camera via:

```
ffmpeg -i http://192.168.0.102:8080/video -f v4l2 -pix_fmt yuyv422 /dev/video4
```

Make sure the camera is streamable via

```
ffplay /dev/video4
```

# VLA

## Env

Reset the environment to switch to Python 3.10

```
deactivate
python_version=3.10.16
export PATH="$HOME/.pyenv/versions/$python_version/bin:$PATH"
python -m venv .venv3.10
. .venv3.10/bin/activate
```

Install dependencies

```
pip install modal
pip install feetech-servo-sdk
pip install git+https://github.com/huggingface/lerobot.git
pip install git+https://github.com/NVIDIA/Isaac-GR00T.git
pip install "numpy<2"
```

```
cd scripts
```

## Policy

```
cd scripts/policy
```

### Gr00t

```
cd gr00t
```

#### Replay episode

Evaluate the camera calibration by replaying an episode from the dataset

```
python scripts/policy/gr00t/eval_gr00t_so100.py --dataset_path ~/Documents/python/robotics/so100_ball_cup --cam_idx 2 --actions_to_execute 748
```

<video controls src="https://github.com/user-attachments/assets/ac5b6dc7-b900-4109-8b2c-068c95ad927e"></video>


#### Train

Train the LeRobot dataset on https://botix.cloud/.

#### Inference server

Start inference server via an unencrypted TCP tunnel in a modal remote function, blocking on `RobotInferenceServer.run`.

```
modal run inference_server.py
```
#### Eval

Evaluate the policy by running a new episode.

Find dynamic `host` and `port` from modal tunnel information displayed while starting the inference server.

```
python eval_gr00t_so100.py --dataset_path ~/Documents/python/robotics/so100_ball_cup --cam_idx 2 --actions_to_execute 40 --action_horizon 16 --use_policy --host r19.modal.host --port 39147 --lang_instruction "pick up the golf ball and place it in the cup" --record_imgs
```

#### Transcode the eval video

```
ffmpeg -pattern_type glob -i 'eval_images/img_*.jpg' -c:v libx264 -pix_fmt yuv420p -y episode.mp4
```


### LeRobot policies

```
cd lerobot
```

#### Train

Configure secrets

```
modal secret create wandb-secret WANDB_API_KEY=...
modal secret create hf-secret HF_TOKEN=...
```

Select policy and dataset

```
policy=act
dataset_repo_id=alexis779/so100_ball_cup2
```

Train the policy on the dataset

```
modal run --detach train_policy.py::train_policy --dataset-repo-id $dataset_repo_id --policy-type $policy
modal run train_policy.py::upload_model --dataset-repo-id $dataset_repo_id --policy-type $policy
```

#### Eval

```
python scripts/policy/lerobot/eval_policy.py --robot_type so100 --policy_type $policy --model_path ~/Documents/python/robotics/so100_ball_cup_act
```

## Docker

Build docker image:

```
docker build -t slobot-genesis .
```

Run docker container. Make sure to enable **DRI** for hardware graphics acceleration.

```
docker run -it --security-opt no-new-privileges=true -p 7860:7860 --device=/dev/dri -v $PWD:/home/user/app slobot-genesis-image
```