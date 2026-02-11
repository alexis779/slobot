## Validation & Calibration

A series of scripts are provided to help with validation and calibration.

### 0. Validate the preset qpos in sim

This validates that the robot is in the targetted position preset from the sim qpos.

```
python scripts/validation/0_validate_sim_qpos.py [middle|zero|rotated|rest]
```

| middle                          | zero                        | rotated                           | rest                           |
| ------------------------------- | --------------------------- | --------------------------------- | ------------------------------ |
| ![middle](images/SimMiddle.png) | ![zero](images/SimZero.png) | ![rotated](images/SimRotated.png) | ![rotated](images/SimRest.png) |

### 1. Validate the middle preset pos

For [motor calibration](https://huggingface.co/docs/lerobot/so101#calibration-video), LeRobot suggests the `middle` position, where all the joints are positioned in the middle of their possible range of movement.

Position the arm manually into the `middle` preset.

```
python scripts/validation/1_calibrate_motor_pos.py middle
```

It will read the motor positions and output them. It should return an int vector around `[2047, 2047, 2047, 2047, 2047, 2047]`, the _middle_ position for each motor.

- For Genesis simulator, the joint position is in _radian_. The _middle_ keyframe corresponds to `[0, -np.pi/2, np.pi/2, 0, 0, -0.15]`. It's `0` reference is defined in the Mujoco configuration, which is the _zero_ keyframe. The `shoulder_pan` joint in the simulator turns in oppposite direction as the Feetech motor, the other ones in the same one, hence the `[-1, 1, 1, 1, 1, 1]` motor direction configuration.

- For the Feetech motor, the motor position range is `[0, 4096[`. The `0` motor position can be offset by storing the homing offset in the Feetech persistent memory. Sending a control command of `2047` to the robot will result in targetting the _middle_ keyframe that was just calibrated.

- For the LeRobot dataset, the joint ranges are `[-100, 100]` except for the gripper, which range is `[0, 100]`. So the instrumented motor values are offset and then scaled to fit in that range, which min is `-100`, middle is `0` and max is `+100`.

#### LeRobot calibration

Running the calibration command will generate the calibration configuration file.

```
lerobot-calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm

```

A calibration sample file `~/.cache/huggingface/lerobot/calibration/robots/so100_follower/follower_arm.json` looks like

```
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 32,
        "range_min": 758,
        "range_max": 3300
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -57,
        "range_min": 736,
        "range_max": 3325
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": 10,
        "range_min": 836,
        "range_max": 3087
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -78,
        "range_min": 803,
        "range_max": 3349
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 30,
        "range_min": 0,
        "range_max": 4095
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -1099,
        "range_min": 2017,
        "range_max": 3470
    }
}
```

### 2. Validate the preset _pos to qpos_ conversion in sim

Same as script 0, but using the motor pos instead of the sim qpos.

```
python scripts/validation/2_validate_sim_pos.py [middle|zero|rotated|rest]
```

### 3. Validate the preset pos in real

Similar than 2 which but now in real. It validates the robot is positioned correctly to the target pos.

```
python scripts/validation/3_validate_real_pos.py [middle|zero|rotated|rest]
```

| middle                           | zero                         | rotated                            | rest                            |
| -------------------------------- | ---------------------------- | ---------------------------------- | ------------------------------- |
| ![middle](images/RealMiddle.png) | ![zero](images/RealZero.png) | ![rotated](images/RealRotated.png) | ![rotated](images/RealRest.png) |

### 4. Validate real to sim

This validates that moving the real robot also updates the rendered robot in sim.

```
python scripts/validation/4_validate_real_to_sim.py [middle|zero|rotated|rest]
```

### 5. Validate sim to real

This validates the robot simulation also controls the physical robot.

```
python scripts/validation/5_validate_sim_to_real.py [middle|zero|rotated|rest]
```
