# Dataset Generation

An alternative to [teleoperation](./teleoperate.md) is to run a scripted policy to generate episodes with random inputs. It creates a synthetic dataset to evaluate other types of policies, for example trained via *Imitation Learning* or *Reinforcement Learning*.

This will generate a list of recordings in the rerun.io format.

```
uv run python scripts/sim/generate_sim_dataset.py --episode-count 40
```

## Rerun to LeRobot conversion

```
uv run python scripts/sim/convert_rerun_to_lerobot_dataset.py --rerun-dir ~/.slobot/recordings --dataset-id alexis779/so100_ball_cup_sim --episode-ids 0
```

Upload the dataset to the HuggingFace hub

```
uv run hf upload alexis779/so100_ball_cup_sim ~/.cache/huggingface/lerobot/alexis779/so100_ball_cup_sim --repo-type dataset
```