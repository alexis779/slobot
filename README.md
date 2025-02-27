# LeRobot SO-ARM-100 6 DOF robotic arm manipulation with Genesis simulator and Feetech motors

There are 2 main use cases
1. sim to real, where genesis controls the physical robot
2. real to sim, where the servo positions will trigger rendering in genesis

## Acknowledgements

### The Robot Studio

[SO-ARM-100](https://github.com/TheRobotStudio/SO-ARM100) provides CAD & [STL model](https://github.com/TheRobotStudio/SO-ARM100/blob/main/stl_files_for_3dprinting/Follower/Print_Follower_SO_ARM100_08k_Ender.STL) of the robotic arm links. After 3D-printing them and ordering the remaining parts, the robot can be assembled for prototyping.

### Servos

[Feetech STS2115](https://www.feetechrc.com/74v-19-kgcm-plastic-case-metal-tooth-magnetic-code-double-axis-ttl-series-steering-gear.html) is the servo inserted in each joint of the robot.

- It's *motor* rotates the connected link.
- It's *magnetic encoder* measures the absolute angular position of the joint.

### LeRobot

[LeRobot](https://github.com/huggingface/lerobot) provides a SOTA library to perform *Imitation Learning* and *Reinforcement Learning*.

Curate of a new dataset: have the follower arm perform a task of interest, by replicating the motion of the leader arm held a human operator.

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

## Setup the environment

```
git clone https://github.com/alexis779/slobot
cd slobot
conda env create -f ./environment.yml
```

## Robot configuration

The example loads the [Mujoco XML configuration](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100).

Make sure to checkout the config repo in the same parent directory.

```
cd ..
git clone https://github.com/google-deepmind/mujoco_menagerie
```

## Validation & Calibration

A series of scripts are provided to help with calibration.

LeRobot suggests 3 keys positions
1. zero
2. rotated
3. rest

### 0. Validate the preset qpos in sim

This validates that the robot is in the targetted position preset in sim.

```
PYOPENGL_PLATFORM=glx python 0_validate_sim_qpos.py [zero|rotated|rest]
```

| zero | rotated | rest |
|----------|-------------|-------|
| ![zero](doc/SimZero.png) | ![rotated](doc/SimRotated.png) | ![rotated](doc/SimRest.png) |


### 1. Calibrate the preset pos

Position the arm manually into the targetted position preset as displayed above. Refer to [LeRobot calibration section](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md#a-manual-calibration-of-follower-arm) and [manual calibration script](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/robots/feetech_calibration.py#L401).

```
python 1_calibrate_motor_pos.py [zero|rotated|rest]
```

### 2. Validate the preset *pos to qpos* conversion in sim

Same as script 0, but using the calibrated motor step positions instead of angular joint positions.

```
PYOPENGL_PLATFORM=glx python 2_validate_sim_pos.py [zero|rotated|rest]
```

### 3. Validate the preset pos in real

Similar than 2 which is in sim but now in real. It validates the robot is positioned correctly to the target pos.

```
python 3_validate_real_pos.py [zero|rotated|rest]
```

### 4. Validate real to sim

This validates that moving the real robot also updates the rendered robot in sim.

```
PYOPENGL_PLATFORM=glx python 4_validate_real_to_sim.py [zero|rotated|rest]
```

### 5. Validate sim to real

This validates the robot simulation also controls the physical robot.

```
PYOPENGL_PLATFORM=glx python 4_validate_real_to_sim.py [zero|rotated|rest]
```


## Examples

### Real

This example moves the robot to the 3 preset positions, waiting 1 sec in between each one.

```
python real.py
```

<video controls><source src="doc/Real3Presets.webm" type="video/webm"></video>


### Sim To Real

This example performs the 3 elemetal rotations in sim and real.
The simulation generates steps, propagating the joint positions to the Feetech motors.

```
python sim_to_real.py
```


| sim | real |
|----------|-------------|
| <video controls><source src="doc/SimToReal_Sim.webm" type="video/webm"></video> | <video controls><source src="doc/SimToReal_Real.webm" type="video/webm"></video> |