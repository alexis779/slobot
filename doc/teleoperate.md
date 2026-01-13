# Tele-operation

Run tele-operation, controlling the follower using the leader position.

```
% python scripts/teleop/teleoperate.py --recording_id episode01 --fps 30
2026-01-13 09:14:13,315 - INFO - Recording /tmp/slobot/teleoperation/episode01.rrd started.
```


# Replay in rerun.io

The robot state for both the leader and the follower can be visualized in rerun.io viewer.

Each joint chart contain 3 metrics
- the leader motor position
- the follower motor position
- the genesis entity qpos

![Recording dashboard](./TeleopRerun.io.png)

```
rerun /tmp/slobot/teleoperation/episode01.rrd
```

# Replay in SIM

Compare the real recording with the SIM replayed videos.

<video controls src="https://github.com/user-attachments/assets/0cd6b8a6-f75c-4e72-adf0-ffdeddc1c45b"></video>



## Visual

<video controls src="https://github.com/user-attachments/assets/ece6a2f4-9c7d-46b6-83bb-ccc295254e9e"></video>

```
python scripts/teleop/replay_recording.py --rrd_file /tmp/slobot/teleoperation/episode01.rrd --fps 30 --substeps 40 --vis_mode visual
```

## Collision

<video controls src="https://github.com/user-attachments/assets/4f362763-4025-4077-8dc4-2f78281f6502"></video>

```
python scripts/teleop/replay_recording.py --rrd_file /tmp/slobot/teleoperation/episode01.rrd --fps 30 --substeps 40 --vis_mode collision
```