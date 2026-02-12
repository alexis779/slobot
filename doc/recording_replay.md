# Replay in rerun.io

The robot state for both the leader and the follower can be visualized in rerun.io viewer.

Each joint chart contain 2 metrics

- the leader motor position
- the follower motor position

![Recording dashboard](./images/TeleopRerun.io.png)

```
rerun ./doc/assets/episode.rrd
```

# Replay in SIM

Compare the real recording with the SIM replayed videos.

<video controls src="https://github.com/user-attachments/assets/0cd6b8a6-f75c-4e72-adf0-ffdeddc1c45b"></video>

## Visual

<video controls src="https://github.com/user-attachments/assets/1bc9a00e-fdda-4590-8fb7-ee414f0ef183"></video>

```
python scripts/teleop/replay_recording.py --rrd_file /tmp/slobot/teleoperation/episode01.rrd --fps 30 --substeps 40 --vis_mode visual
```

## Collision

<video controls src="https://github.com/user-attachments/assets/0e8e0346-5ef1-475e-9eba-1374347e4f71"></video>

```
python scripts/teleop/replay_recording.py --rrd_file /tmp/slobot/teleoperation/episode01.rrd --fps 30 --substeps 40 --vis_mode collision
```
