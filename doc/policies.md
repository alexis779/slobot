## Policies

### Scripted policy

This policy relies on Inverse Kinematics + Path Planning to perform the task.

```
python scripts/validation/run_sim_policy.py --ball_x -5 --ball_y -11 --cup_x 8 --cup_y -11
```

<video controls src="https://github.com/user-attachments/assets/14c96939-5c3c-46ea-9dce-c858aae510a9"></video>

### ACT

Action Chunking Transformer is part of the [Aloha paper](https://tonyzhaozh.github.io/aloha/).

### PI0

PI0 was released by Physical Intelligence to provide an OpenSource policy similar to Figure AI Helix architecture.

### Isaac Gr00t N1

[NVIDIA Isaac GR00T N1](https://github.com/NVIDIA/Isaac-GR00T) was released by Nvidia.

### SmolVLA

SmolVLA was released by HuggingFace, using LeRobot datasets for training.

# VLA

## Env

Reset the environment to switch to Python 3.10

```
deactivate
python_version=3.10.16
export PATH="$HOME/.pyenv/versions/$python_version/bin:$PATH"
python -m venv .venv3.10
. .venv3.10/bin/activate
```

Install dependencies

```
pip install modal
pip install feetech-servo-sdk
pip install git+https://github.com/huggingface/lerobot.git
pip install git+https://github.com/NVIDIA/Isaac-GR00T.git
pip install "numpy<2"
```

```
cd scripts
```

## Policy

```
cd scripts/policy
```

### Gr00t

```
cd gr00t
```

#### Replay episode

Evaluate the camera calibration by replaying an episode from the dataset

```
python scripts/policy/gr00t/eval_gr00t_so100.py --dataset_path ~/Documents/python/robotics/so100_ball_cup --cam_idx 2 --actions_to_execute 748
```

<video controls src="https://github.com/user-attachments/assets/ac5b6dc7-b900-4109-8b2c-068c95ad927e"></video>


#### Train

Train the LeRobot dataset on https://botix.cloud/.

#### Inference server

Start inference server via an unencrypted TCP tunnel in a modal remote function, blocking on `RobotInferenceServer.run`.

```
modal run inference_server.py
```
#### Eval

Evaluate the policy by running a new episode.

Find dynamic `host` and `port` from modal tunnel information displayed while starting the inference server.

```
python eval_gr00t_so100.py --dataset_path ~/Documents/python/robotics/so100_ball_cup --cam_idx 2 --actions_to_execute 40 --action_horizon 16 --use_policy --host r19.modal.host --port 39147 --lang_instruction "pick up the golf ball and place it in the cup" --record_imgs
```

#### Transcode the eval video

```
ffmpeg -pattern_type glob -i 'eval_images/img_*.jpg' -c:v libx264 -pix_fmt yuv420p -y episode.mp4
```


### LeRobot policies

```
cd lerobot
```

#### Train

Configure secrets

```
modal secret create wandb-secret WANDB_API_KEY=...
modal secret create hf-secret HF_TOKEN=...
```

Select policy and dataset

```
policy=act
dataset_repo_id=alexis779/so100_ball_cup2
```

Train the policy on the dataset

```
modal run --detach train_policy.py::train_policy --dataset-repo-id $dataset_repo_id --policy-type $policy
modal run train_policy.py::upload_model --dataset-repo-id $dataset_repo_id --policy-type $policy
```

#### Eval

```
python scripts/policy/lerobot/eval_policy.py --robot_type so100 --policy_type $policy --model_path ~/Documents/python/robotics/so100_ball_cup_act
```
