Refer to the [HuggingFace tutorial](https://huggingface.co/docs/lerobot/en/hilserl).

# Calibrate the follower and leader

Pass supported `so100_follower` **robot.type** and `so100_leader` **teleop.type**.

This calibration ensures the middle motor position (`motor step = 4096 / 2 = 2048`) matches exactly the joint range middle point.

## Follower

```bash
uv run lerobot-calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm
```

## Leader

```bash
uv run lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm
```

# Verify cameras

Make sure cameras are available:

```bash
$ v4l2-ctl --list-devices
HD Webcam eMeet C950: HD Webcam (usb-0000:03:00.3-1.4.2):
        /dev/video4
        /dev/video5
        /dev/media2

USB CAMERA: USB CAMERA (usb-0000:03:00.3-1.4.3):
        /dev/video2
        /dev/video3
        /dev/media1
```

# Teleoperate the robot

Make sure tele-operation works:


```bash
uv run lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
```

Make sure tele-operation with live simulation works:

```bash
uv run slobot-teleoperate \
    --robot.type=slobot_so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=slobot_so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=slobot_leader_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
```

# Record the dataset

Choose a dataset id for offline policy learning, for example `alexis779/so100_cube_rectangle_day`.

Human reward labels (leader GUI): **s** when the cube is on the rectangle (reward for
`env.fps` control frames, i.e. 1 second at the control rate), move the arm to rest,
**q** to end the episode. **r** to rerecord.
`terminate_on_success` is false so the episode continues after **s**. Set
`reward_classifier.pretrained_path` to `null` during recording so labels come only from teleop.

Leader arm joints are FK’d to gripper_link pose; follower jaw uses present follower motor
position (not the leader gripper).

- `observation.state` / `action`: `[x, y, z, r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, jaw]` — gripper_link position (m), 6D rotation (first two columns of the rotation matrix), jaw motor position (rad). Action dim 9 uses the same jaw motor position representation as `observation.state` (not discrete gripper commands). Press **o** / **c** in the leader GUI to open/close the follower jaw (leader jaw position is not forwarded).

- Leader teleop: arm joints from the leader; follower jaw from present follower motors. Policy path: Genesis sim collision check before sending motor goals; invalid poses get reward penalty `-1` and the arm does not move.

```
uv run slobot-record --config_path ./src/slobot/hilserl/configs/record_dataset_config.json
```

Delete previously recorded dataset when re-recording it.

```
rm -r ~/.cache/huggingface/lerobot/alexis779/so100_cube_rectangle_day
```

# Visualize a dataset episode

```
uv run lerobot-dataset-viz --root ~/.cache/huggingface/lerobot/alexis779/so100_cube_rectangle_day --repo-id alexis779/so100_cube_rectangle_day --episode-index 0
```

<video controls src="https://github.com/user-attachments/assets/204a97bb-bfe4-4e6b-8749-2ed07ccce6e7">
</video>

# Train the reward classifier

This will train the reward classifier on the CPU.

```
uv run lerobot-train \
    --config_path ./src/slobot/hilserl/configs/train_reward_classifier_config.json
```

![Training Loss](./images/hilserl/RewardClassifierTrainingLoss.png)

## On Modal

```bash
uv run modal run src/slobot/hilserl/modal/train_reward_classifier.py
```

Checkpoints are written to the Modal volume under `outputs/train/{date}/{time}_reward-classifier/`.


# Evaluate the reward classifier

Edit the record config with the following changes

```json
+++ b/src/slobot/hilserl/configs/record_dataset_config.json
@@ -68,16 +68,16 @@
                 }
             },
             "reward_classifier": {
-                "pretrained_path": null,
+                "pretrained_path": "alexis779/so100_cube_rectangle_day_reward_classifier",
                 "success_threshold": 0.5,
                 "success_reward": 1.0
             }
         }
     },
     "dataset": {
-        "repo_id": "alexis779/so100_cube_rectangle_day",
+        "repo_id": "alexis779/so100_cube_rectangle_day_eval",
         "task": "Pick cube and place it on the rectangle",
-        "num_episodes_to_record": 10,
+        "num_episodes_to_record": 3,
         "push_to_hub": true
     },
     "mode": "record",
```


Record an eval dataset, but without pressing the s key.
```
uv run slobot-record --config_path ./src/slobot/hilserl/configs/record_dataset_config.json
```

It should detect task-successful frames automatically, without the operator pressing s key while tele-operating. Each frame also records `next.reward.probability` the classifier float value.


```
uv run lerobot-dataset-viz --root ~/.cache/huggingface/lerobot/alexis779/so100_cube_rectangle_day_eval --repo-id alexis779/so100_cube_rectangle_day_eval --episode-index 0
```

![Reward Classification](./images/hilserl/RewardClassificationEpisodeEval.png)

# Run Online RL

`train_hil_serl_config.json` shares the same robot, cameras, URDF, reset pose, and **leader teleop** as recording.

Online RL starts in **policy mode** (`IS_INTERVENTION: false`). The follower tracks the policy EE pose via IK. Press **i** in the leader GUI to take over (leader joints → clipped EE → follower joints); press **i** again to return control to the policy. Keep the teleop window focused so keypresses are received. Recording uses `default_intervention: true` so demos start in teleop mode.

- `observation.state` / `action`: normalized `[x, y, z, r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, jaw]` (10D), same min/max as `dataset_stats`. Keys **o** / **c** open/close gripper during teleop.

## Learner

Start the learner locally:

```
uv run slobot-learner --config_path ./src/slobot/hilserl/configs/train_hil_serl_config.json
```

### On Modal

Run the **learner** on [Modal](https://modal.com/) (GPU in the cloud) while the **actor** stays on the computer next to the robot.

Modal exposes gRPC with an **unencrypted TCP tunnel** (`modal.forward`), which matches LeRobot’s `grpc.insecure_channel` on the actor.

**Why GPU for the learner?** The actor only runs policy inference at `env.fps` (10 Hz) on CPU. The learner is different: each training step runs SAC with a ResNet10 vision encoder, batch size 256, and `utd_ratio: 2` (two critic updates per step). On CPU that work is roughly **0.066 Hz** — about **15 s per optimization step** — because almost all time is spent in forward/backward passes over image observations. Modal gives the learner a GPU (`policy.device=cuda`) so those passes are much faster; `policy.storage_device=cpu` keeps the replay buffers on CPU and only moves batches to the GPU for training. The actor is unaffected by learner throughput: it keeps collecting transitions at 10 Hz while the learner trains asynchronously and pushes updated weights every few seconds.

Start the remote learner

```bash
uv run modal run src/slobot/hilserl/modal/learner.py
```

In the Modal logs, look for:

```text
Connect the actor (computer) with:
  --policy.actor_learner_config.learner_host=<modal-host>
  --policy.actor_learner_config.learner_port=<modal-port>
```

## Actor

Use the **same** `train_hil_serl_config.json` as locally, but point at the tunnel:

```bash
uv run slobot-actor --config_path ./src/slobot/hilserl/configs/train_hil_serl_config.json \
    --policy.device=cpu \
    --policy.actor_learner_config.learner_host=<modal-host> \
    --policy.actor_learner_config.learner_port=<modal-port>
```
