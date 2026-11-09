## Wrist roll sweep

Place the arm in the `center` preset, park `Wrist_Roll` at its lower limit, and sweep it toward the upper limit while recording the mobile camera. The sweep stays inside the joint limits, about **-2.79** to **2.79** rad.

An orange sphere is placed on the plane at the fixed jaw's (x, y). Inverse kinematics keeps that jaw position and turns the jaw so the mobile camera looks at the sphere. The Genesis viewer stays open for the sweep, then the script exits.

```
uv run python scripts/record_wrist_roll_mobile.py
```

The sim recording is written to `~/.slobot/wrist_roll_center_sim.mp4`.

### Real arm and camera

`--feetech` commands the follower arm from the sim joint position on every step. `--video-id` is the V4L2 device index, so `2` records `/dev/video2` for the same sweep.

```
uv run python scripts/record_wrist_roll_mobile.py --feetech --video-id 2
```

| real                                                                                                           | sim                                                                                                            |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <video controls src="https://github.com/user-attachments/assets/943f79a6-8c7c-48d8-8f12-d8b2f2c923b4"></video> | <video controls src="https://github.com/user-attachments/assets/3306c2ae-d92a-48fb-9baa-d5fc1387b0c3"></video> |

| Camera | File |
| --- | --- |
| Sim mobile camera | `~/.slobot/wrist_roll_center_sim.mp4` |
| `/dev/video2` | `~/.slobot/wrist_roll_center_real.mp4` |

`--video-id` is required with `--feetech`. After the sweep, the real arm moves to the `rest` preset and torque is turned off.
