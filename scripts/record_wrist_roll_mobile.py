"""Place the arm in the center preset and sweep Wrist_Roll while recording the mobile camera.

Pass --feetech to command the real follower arm from the sim joint positions.
Pass --video-id 2 to also record /dev/video2 during the sweep.
"""

import argparse
import math
import signal
import subprocess
from pathlib import Path

import genesis as gs
import torch
from genesis.utils import geom as gu

from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100

WRIST_JOINT = "Wrist_Roll"
PRESET = "center"
FPS = 30
SWEEP_SECONDS = 6.0
HOLD_SECONDS = 0.5
SETTLE_STEPS = 10
MARKER_RADIUS = 0.02
SIM_OUTPUT = Path(Configuration.WORK_DIR) / "wrist_roll_center_sim.mp4"
REAL_OUTPUT = Path(Configuration.WORK_DIR) / "wrist_roll_center_real.mp4"


def main():
    args = _parse_args()
    feetech = Feetech(qpos_map=Configuration.MJCF_QPOS_MAP) if args.feetech else None
    real_recording = [None]
    try:
        _record(feetech, args.video_id, real_recording)
    finally:
        _stop_real_recording(real_recording[0])
        if feetech is not None:
            feetech.go_to_rest()


def _parse_args():
    parser = argparse.ArgumentParser(description="Sweep Wrist_Roll and record the mobile camera.")
    parser.add_argument(
        "--video-id",
        type=int,
        help="V4L2 device index to record during the sweep, for example 2 for /dev/video2.",
    )
    parser.add_argument(
        "--feetech",
        action="store_true",
        help="Command the real follower arm from the sim joint positions.",
    )
    args = parser.parse_args()
    if args.feetech and args.video_id is None:
        parser.error("--video-id is required with --feetech")
    return args


def _camera_device(video_id: int) -> str:
    return f"/dev/video{video_id}"


def _start_real_recording(video_id: int, path: Path) -> subprocess.Popen:
    device = _camera_device(video_id)
    if not Path(device).exists():
        raise RuntimeError(f"{device} is not available")
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", "yuyv422",
            "-video_size", "640x480",
            "-framerate", str(FPS),
            "-i", device,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
    )


def _stop_real_recording(process: subprocess.Popen | None):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    process.wait(timeout=15)


def _record(feetech, video_id: int | None, real_recording: list):
    arm = SoArm100(show_viewer=True, fps=FPS, should_start=False, step_handler=feetech)
    genesis = arm.genesis
    genesis.start()
    marker = genesis.scene.add_entity(
        gs.morphs.Sphere(radius=MARKER_RADIUS, fixed=True, collision=False),
        surface=gs.surfaces.Rough(color=(0.95, 0.25, 0.05)),
        visualize_contact=False,
    )
    genesis.build()
    # The marker sits just under the jaw, closer than the default 0.1 m near plane.
    genesis.link_camera._rasterizer._camera_nodes[genesis.link_camera.uid].camera.znear = 0.01
    entity = genesis.entity

    preset = torch.tensor(arm.preset_qpos(PRESET), dtype=torch.float32)
    # idx_local is the qpos / dof index. get_qpos_idx subtracts 1 for a base joint this MJCF does not have.
    wrist_idx = int(entity.get_joint(WRIST_JOINT).idx_local)
    qmin, qmax = _dof_limits(entity.get_dofs_limit(), wrist_idx)

    qpos = preset.clone()
    _command(entity, qpos)
    for _ in range(SETTLE_STEPS):
        genesis.step()

    qpos[wrist_idx] = qmin
    _command(entity, qpos)
    for _ in range(SETTLE_STEPS):
        genesis.step()

    jaw_pos = _link_pos(genesis.link)
    _place_on_plane(marker, jaw_pos)
    qpos = _aim_jaw_at_marker(entity, genesis.link, arm.camera_offset(), qpos, jaw_pos, wrist_idx)
    qpos[wrist_idx] = qmin
    _command(entity, qpos)
    for _ in range(SETTLE_STEPS):
        genesis.step()

    jaw_pos = _link_pos(genesis.link)
    marker_pos = _place_on_plane(marker, jaw_pos)
    print(f"fixed jaw={_fmt(jaw_pos)} marker={_fmt(marker_pos)}")

    span = min(2 * math.pi, qmax - qmin)
    target = qmin + span
    print(f"Wrist_Roll limits=[{qmin:.4f}, {qmax:.4f}] sweep_to={target:.4f}")

    SIM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if video_id is not None:
        real_recording[0] = _start_real_recording(video_id, REAL_OUTPUT)
    genesis.link_camera.start_recording(save_to_filename=str(SIM_OUTPUT), fps=FPS)

    n_hold = max(1, round(HOLD_SECONDS * FPS))
    n_sweep = max(2, round(SWEEP_SECONDS * FPS))

    def pose(angle: float):
        commanded = qpos.clone()
        commanded[wrist_idx] = min(max(angle, qmin), qmax)
        _command(entity, commanded)
        genesis.step()

    for _ in range(n_hold):
        pose(qmin)
    for angle in torch.linspace(qmin, target, n_sweep):
        pose(float(angle))
    for _ in range(n_hold):
        pose(target)

    genesis.link_camera.stop_recording()
    _stop_real_recording(real_recording[0])
    print(SIM_OUTPUT.resolve())
    if video_id is not None:
        print(REAL_OUTPUT.resolve())
    genesis.stop()


def _aim_jaw_at_marker(entity, link, camera_T, qpos, jaw_pos, wrist_idx):
    """IK the fixed jaw so the mobile camera looks at the marker, keeping wrist roll and jaw position."""
    marker_pos = _marker_position(jaw_pos)
    quat = _link_quat(link)
    rotation = _aim_rotation(jaw_pos, quat, camera_T, marker_pos)
    target_quat = gu.R_to_quat(rotation)
    other_dofs = [index for index in range(int(qpos.numel())) if index != wrist_idx]
    solved = entity.inverse_kinematics(
        link=link,
        pos=jaw_pos.unsqueeze(0),
        quat=target_quat.reshape(1, 4),
        init_qpos=qpos,
        respect_joint_limit=True,
        dofs_idx_local=other_dofs,
    )
    solved = solved.reshape(-1)[: qpos.numel()].to(dtype=qpos.dtype)
    lower, upper = entity.get_dofs_limit()
    lower = torch.as_tensor(lower, dtype=solved.dtype).reshape(-1)
    upper = torch.as_tensor(upper, dtype=solved.dtype).reshape(-1)
    return torch.clamp(solved, lower, upper)


def _aim_rotation(link_pos, link_quat, camera_T, target):
    rotation = gu.quat_to_R(link_quat.reshape(1, 4))[0]
    camera_T = torch.as_tensor(camera_T, dtype=rotation.dtype, device=rotation.device)
    anchor = camera_T[:3, 3]
    # Genesis cameras look along -Z of the offset rotation.
    aim_local = -camera_T[:3, 2]
    aim_local = aim_local / torch.linalg.norm(aim_local)
    for _ in range(4):
        camera_pos = link_pos + rotation @ anchor
        desired = target - camera_pos
        desired = desired / torch.linalg.norm(desired)
        current = rotation @ aim_local
        rotation = _rotation_between(current, desired) @ rotation
    return rotation


def _rotation_between(src, dst):
    src = src / torch.linalg.norm(src)
    dst = dst / torch.linalg.norm(dst)
    cross = torch.cross(src, dst, dim=0)
    cosine = torch.dot(src, dst).clamp(-1.0, 1.0)
    sine_norm = torch.linalg.norm(cross)
    if sine_norm < 1e-8:
        if cosine > 0:
            return torch.eye(3, dtype=src.dtype, device=src.device)
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=src.dtype, device=src.device)
        if abs(float(src[0])) > 0.9:
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=src.dtype, device=src.device)
        axis = torch.cross(src, axis, dim=0)
        axis = axis / torch.linalg.norm(axis)
        return 2.0 * torch.outer(axis, axis) - torch.eye(3, dtype=src.dtype, device=src.device)

    axis = cross / sine_norm
    angle = torch.acos(cosine)
    skew = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=src.dtype,
        device=src.device,
    )
    eye = torch.eye(3, dtype=src.dtype, device=src.device)
    return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _place_on_plane(marker, jaw_pos):
    marker_pos = _marker_position(jaw_pos)
    marker.set_pos(marker_pos.unsqueeze(0))
    return marker_pos


def _marker_position(jaw_pos):
    position = jaw_pos.detach().clone()
    position[2] = MARKER_RADIUS
    return position


def _link_pos(link):
    return link.get_pos().reshape(-1)[:3]


def _link_quat(link):
    return link.get_quat().reshape(-1)[:4]


def _command(entity, qpos: torch.Tensor):
    entity.control_dofs_position(qpos)
    entity.set_dofs_position(qpos)


def _dof_limits(limits, index: int) -> tuple[float, float]:
    lower, upper = limits
    lower = torch.as_tensor(lower).reshape(-1)
    upper = torch.as_tensor(upper).reshape(-1)
    return float(lower[index]), float(upper[index])


def _fmt(vector) -> str:
    values = [float(v) for v in vector.reshape(-1)[:3]]
    return "(" + ", ".join(f"{v:.4f}" for v in values) + ")"


if __name__ == "__main__":
    main()
