# Installation

## Python venv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

Create virtual environment:

```
uv venv
```

## Dependencies

Install following dependencies

### 1. slobot

```
uv pip install -e .
```

### 2. Robot Configuration

#### URDF

Ensure the URDF [configuration](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO100/so100.urdf) in available in `slobot.config` package.

```
cd ..
git clone https://github.com/TheRobotStudio/SO-ARM100
cd slobot
ln -s `pwd`/../SO-ARM100/Simulation/SO100 src/slobot/config/SO100
```

#### MJCF

Ensure the Mujoco [configuration](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) in available in `slobot.config` package.

```
cd ..
git clone https://github.com/google-deepmind/mujoco_menagerie
cd slobot
ln -s `pwd`/../mujoco_menagerie/trs_so_arm100 src/slobot/config/trs_so_arm100
```

### 3. LeRobot

```
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git

uv pip install 'lerobot[dataset]'
uv pip install 'lerobot[feetech]'

uv pip install 'lerobot[training]'
uv pip install transformers

uv pip uninstall opencv-python-headless
uv pip install --reinstall opencv-python
uv pip install torchcodec
```

### 4. Genesis

```
uv pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

Also refer to the [installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html). Make sure to run the [hello world example](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html) successfully.

## Docker


### Local

Build docker image:

```
docker build -f docker/Dockerfile.local -t slobot .
```

Run docker container. Make sure to enable **DRI** for hardware graphics acceleration.

```
docker run -it --security-opt no-new-privileges=true -p 7860:7860 --device=/dev/dri -v $PWD:/home/user/app slobot
```