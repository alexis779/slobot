"""X11 / GUI initialization for Tk + Genesis + OpenCV in one process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.envs import HILSerlRobotEnvConfig


def init_x11_threading() -> None:
    """Enable threaded X11 access before Tk, OpenCV, or Genesis open a display."""
    import ctypes

    for lib_name in ("libX11.so.6", "libX11.so"):
        try:
            ctypes.CDLL(lib_name).XInitThreads()
            return
        except OSError:
            continue


_opencv_gui_ready = False


def init_opencv_gui() -> None:
    """Prepare OpenCV highgui for use alongside Tk on the same X display."""
    global _opencv_gui_ready
    if _opencv_gui_ready:
        return
    import cv2

    cv2.startWindowThread()
    _opencv_gui_ready = True


def make_robot_env_with_genesis_before_tk(cfg: HILSerlRobotEnvConfig, *, factory: Any):
    """Connect follower (Genesis viewer) before opening the leader Tk window."""
    from lerobot.rl.gym_manipulator import RobotEnv

    assert cfg.robot is not None
    assert cfg.teleop is not None

    robot = factory.make_robot_from_config(cfg.robot)
    teleop_device = factory.make_teleoperator_from_config(cfg.teleop)

    processor = cfg.processor
    use_gripper = processor.gripper.use_gripper if processor.gripper is not None else True
    display_cameras = (
        processor.observation.display_cameras if processor.observation is not None else False
    )
    reset_pose = processor.reset.fixed_reset_joint_positions if processor.reset is not None else None

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    if display_cameras:
        init_opencv_gui()

    teleop_device.connect()
    return env, teleop_device
