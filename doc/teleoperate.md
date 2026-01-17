# Tele-operation

Run tele-operation, controlling the follower using the leader position.

## Architecture

The teleoperation uses an asynchronous architecture with Linux FIFO queues for Inter-Process Communication. It sends the data and metrics to Rerun.io as a sink.

| Resource     | Repeated Action |
|--------------|-----------------|
| Leader Arm   | Read leader arm motor positions |
| Follower Arm| Send control command to the follower arm and read its motor positions |
| Webcam       | Capture an image of the scene |
| Simulator    | Step through the simulation |
| Cron         | Send periodic tick |


### Leader Read

```
python scripts/teleop/asyncprocessing/spawn_leader_read.py --recording-id episode --port /dev/ttyACM1
```


### Follower Control

```
python scripts/teleop/asyncprocessing/spawn_follower_control.py --recording-id episode --port /dev/ttyACM0 --webcam --sim
```


### Webcam Capture
```
python scripts/teleop/asyncprocessing/spawn_webcam_capture.py --recording-id episode --camera-id 2 --width 640 --height 480 --fps 30
```


### Sim Step
```
python scripts/teleop/asyncprocessing/spawn_sim_step.py --recording-id episode --width 640 --height 480 --fps 30 --substeps 40 --vis-mode visual
```

### Cron loop

```
python scripts/teleop/asyncprocessing/spawn_cron.py --recording-id episode --fps 30
```