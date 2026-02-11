## Installation

### Python

Create a virtual environment with _venv_

```
sudo apt install python3.13-venv
```

```
python3.13 -m venv .venv
. .venv/bin/activate
```

Validate python version in the venv

```
(.venv) $ python -V
Python 3.13.5
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

On Ubuntu, Qt5 library may be incompatible with [pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) native library. See [reported issue](https://github.com/Genesis-Embodied-AI/Genesis/issues/189). As a workaround, give precedence to the _python module_ QT library instead of the _Ubuntu system_ QT library.

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
LD_LIBRARY_PATH=$PYMESHLAB_LIB python <script.py>
```

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