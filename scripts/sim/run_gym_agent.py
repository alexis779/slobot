import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from slobot.sim.gym.gym_env import GymEnv
from slobot.sim.recording_layout import RecordingLayout

gym_id = "slobot/GolfBallEnv-v0"
gym.register(
    id=gym_id,
    entry_point=GymEnv,
    max_episode_steps=100,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run random-action Gym agent on GolfBallEnv"
    )
    parser.add_argument(
        "--ball-x",
        type=float,
        required=True,
        help="Golf ball X position in inches",
    )
    parser.add_argument(
        "--ball-y",
        type=float,
        required=True,
        help="Golf ball Y position in inches",
    )
    parser.add_argument(
        "--cup-x",
        type=float,
        required=True,
        help="Cup X position in inches",
    )
    parser.add_argument(
        "--cup-y",
        type=float,
        required=True,
        help="Cup Y position in inches",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        required=True,
        help="Genesis simulation substeps per agent step",
    )
    args = parser.parse_args()

    env = gym.make(gym_id)

    recording_layout = RecordingLayout(
        rrd_file=None,
        pick_frame_id=None,
        pre_grasp_mode=None,
        ball_x=args.ball_x,
        ball_y=args.ball_y,
        cup_x=args.cup_x,
        cup_y=args.cup_y,
        recording_id="gym-agent",
    )

    options = {
        "recording_layout": recording_layout,
        "sim_steps": args.sim_steps,
    }
    obs, _reset_info = env.reset(options=options)

    # TODO change random sample to actual policy output
    action = env.action_space.sample()

    obs, _reward, _terminated, _truncated, _step_info = env.step(action)

    side = "gym_agent_side_camera.png"
    link = "gym_agent_link_camera.png"
    Image.fromarray(np.asarray(obs["side_camera_image"])).save(side)
    Image.fromarray(np.asarray(obs["link_camera_image"])).save(link)
    print(Path(side).resolve())
    print(Path(link).resolve())

    env.close()


if __name__ == "__main__":
    main()
