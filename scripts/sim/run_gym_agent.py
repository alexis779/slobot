import argparse
import statistics
import time
from pathlib import Path

import gymnasium as gym

from slobot.sim.gym.gym_env import GymEnv
from slobot.sim.jepa_policy import JepaPolicy
from slobot.sim.recording_layout import RecordingLayout


def _default_so100_jepa_ac_model_dir() -> Path:
    """../so100-jepa-ac-model/ relative to the slobot repo root."""
    slobot_repo_root = Path(__file__).resolve().parents[2]
    return (slobot_repo_root / ".." / "so100-jepa-ac-model").resolve()


_DEFAULT_ENCODER = _default_so100_jepa_ac_model_dir() / "vision_encoder.pt"
_DEFAULT_PREDICTOR = _default_so100_jepa_ac_model_dir() / "latest.pt"

gym_id = "slobot/GolfBallEnv-v0"
gym.register(
    id=gym_id,
    entry_point=GymEnv,
    max_episode_steps=100,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JEPA policy on GolfBallEnv for a fixed number of steps"
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
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of policy steps (env.step calls) per run",
    )
    parser.add_argument(
        "--encoder-ckpt",
        type=Path,
        default=_DEFAULT_ENCODER,
        help=f"ViT-L encoder checkpoint (default: {_DEFAULT_ENCODER})",
    )
    parser.add_argument(
        "--predictor-ckpt",
        type=Path,
        default=_DEFAULT_PREDICTOR,
        help=f"Predictor checkpoint (default: {_DEFAULT_PREDICTOR})",
    )
    parser.add_argument(
        "--camera",
        choices=("side", "link"),
        default="side",
        help="Which RGB observation matches training (dataset used a single phone view; try side first)",
    )
    args = parser.parse_args()

    encoder_ckpt = args.encoder_ckpt
    predictor_ckpt = args.predictor_ckpt

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

    # gym.make applies a TimeLimit wrapper when max_episode_steps is set on register;
    # .unwrapped is the inner GymEnv with .golf_ball_env.
    gym_env: GymEnv = env.unwrapped
    policy = JepaPolicy(
        gym_env.golf_ball_env,
        encoder_ckpt,
        predictor_ckpt,
        camera=args.camera,
    )
    inference_s: list[float] = []
    for step_idx in range(args.steps):
        t0 = time.perf_counter()
        action = policy.act(obs)
        dt_s = time.perf_counter() - t0
        inference_s.append(dt_s)
        print(
            f"[gym_agent] inference step={step_idx} latency={dt_s * 1e3:.2f} ms",
            flush=True,
        )
        obs, _reward, terminated, truncated, _step_info = env.step(action)
        if terminated or truncated:
            break

    n = len(inference_s)
    if n:
        total_s = sum(inference_s)
        print(
            f"[gym_agent] inferences={n} "
            f"latency_total={total_s * 1e3:.2f} ms "
            f"mean={statistics.mean(inference_s) * 1e3:.2f} ms "
            f"min={min(inference_s) * 1e3:.2f} ms "
            f"max={max(inference_s) * 1e3:.2f} ms",
            flush=True,
        )
    else:
        print("[gym_agent] inferences=0", flush=True)

    env.close()


if __name__ == "__main__":
    main()
