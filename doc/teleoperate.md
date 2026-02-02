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
python scripts/teleop/asyncprocessing/spawn_leader_read.py --port /dev/ttyACM1
```


### Follower Control


```
python scripts/teleop/asyncprocessing/spawn_follower_control.py --port /dev/ttyACM0 --camera-id 2 --camera-id 4 --sim
```


### Webcam Capture

Start the capture from the *fixed webcam* or the *wrist webcam*

```
python scripts/teleop/asyncprocessing/spawn_webcam_capture1.py --camera-id 1 --width 640 --height 480 --fps 30 --detect-objects
```


### Detect Objects
Start object detection via YOLO model. Webcam Capture worker is a producer. Detect Objects worker is a consumer. It reads the image from a shared memory between the two workers.

```
python scripts/teleop/asyncprocessing/spawn_detect_objects.py --camera-id 1 --width 640 --height 480 --detection-task DETECT
```

### Sim Step
```
python scripts/teleop/asyncprocessing/spawn_sim_step.py --width 640 --height 480 --fps 30 --substeps 40 --vis-mode visual
```

### Cron loop

It should be started after spawning all the remaining workers, because it needs to send a special message to configure the workers with a common recording id.

Pass the *recording id* as a command line argument to send the metrics to Rerun.io.

```
python scripts/teleop/asyncprocessing/spawn_cron.py --recording-id episode --fps 30
```


## Teleoperate another robot

This shows how a cheap robot leader arm like SO-ARM-100 can control an industrial robot like Franka arm.

<video controls src="https://github.com/user-attachments/assets/a7a7ba47-17be-46c0-b91b-39a1aa89bd0c"></video>


Start the *Mirror Kinematics* worker instead of *Follower Control* worker:

```
python scripts/teleop/asyncprocessing/spawn_mirror_kinematics.py --fps 30 --substeps 40 --width 640 --height 480 --vis-mode visual --mjcf-path ../mujoco_menagerie/franka_emika_panda/panda.xml --end-effector-link link7
```