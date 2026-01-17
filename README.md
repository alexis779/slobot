# LeRobot SO-ARM-100 6 DOF robotic arm manipulation with Genesis simulator and Feetech motors

There are 2 main use cases

1. sim to real, where genesis controls the physical robot
2. real to sim, where the physical robot moves will refresh the robot rendering in genesis

## TOC

- [Acknowledgements](doc/acknowledgements.md)
- [Installation](doc/installation.md)
- [Calibration & Validation](doc/calibration_validation.md)
- [Real 2 Sim](doc/real2sim.md)
- [Policies](doc/policies.md)
- [Tele-operation](doc/teleoperate.md)
- [Examples](doc/examples.md)
- [Gradio apps](doc/gradio_apps.md)
- [LeRobot dataset EDA](notebooks/so100_ball_cup2.ipynb)

## Goal

The goal is to replay a recorded LeRobot dataset episode or Rerun.io recording in a simulation environment.

| Real | Sim Visual | Sim Collision |
|------|------------|---------------|
| <video controls src="https://github.com/user-attachments/assets/0cd6b8a6-f75c-4e72-adf0-ffdeddc1c45b"></video> | <video controls src="https://github.com/user-attachments/assets/1bc9a00e-fdda-4590-8fb7-ee414f0ef183"></video> | <video controls src="https://github.com/user-attachments/assets/0e8e0346-5ef1-475e-9eba-1374347e4f71"></video> |

## Tele-operation

For live simulation, the tele-operation process should rely on a scalable asynchronous process.

<video controls src="https://github.com/user-attachments/assets/65dc44ab-3c72-4925-8f55-30a4bbf3d3f1"></video>

With pub/sub, workers repeatedly poll tasks from their dedicated queue, pushing their output to rerun.io database.

See [Tele-operation](doc/teleoperate.md) for more details.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e1f5fe', 'secondaryColor': '#fff3e0', 'tertiaryColor': '#f3e5f5'}}}%%
flowchart LR
    subgraph main [Cron]
        Cron["Infinite Loop - 30 Hz"]
    end

    subgraph resources [Resources]
        LeaderArm[Leader Arm]
        FollowerArm[Follower Arm]
        Webcam[Webcam]
        Genesis[Genesis]
    end

    subgraph workers [Worker Processes]
        LeaderRead[Leader Read]
        FollowerControl[Follower Control]
        WebcamCapture[Webcam Capture]
        SimStep[Sim Step]
    end

    subgraph fifos [FIFO Queues]
        Q1([leader_read_q])
        Q2([follower_control_q])
        Q3([sim_step_q])
        Q4([webcam_capture_q])
    end

    subgraph metrics [Database]
        Rerun[(Rerun.io)]
    end

    Cron -->|"empty"| Q1
    Q1 --> LeaderRead
    LeaderRead -->|"qpos[]"| Q2
    Q2 --> FollowerControl
    FollowerControl -->|"empty"| Q4
    Q4 --> WebcamCapture
    FollowerControl -->|"qpos[]"| Q3
    Q3 --> SimStep

    LeaderRead <-->|"qpos[]"| LeaderArm
    FollowerControl <-->|"qpos[]"| FollowerArm
    WebcamCapture <-->|"BGR[][]"| Webcam
    SimStep <-->|"RGB[][]"| Genesis

    Cron --> Rerun
    LeaderRead --> Rerun
    FollowerControl --> Rerun
    WebcamCapture --> Rerun
    SimStep --> Rerun
```