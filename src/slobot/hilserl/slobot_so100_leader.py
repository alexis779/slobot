"""SO-100 leader with slobot joint actions and HIL-SERL teleop events."""

from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.hilserl.config_slobot_so100_leader import SlobotSO100LeaderTeleopConfig
from slobot.hilserl.factory import Factory
from slobot.hilserl.hilserl_gui_state import (
    HilSerlGuiContext,
    HilSerlGuiMode,
    HilSerlGuiStateMachine,
)
from slobot.hilserl.models.gripper_command import GripperCommand

logger = logging.getLogger(__name__)

_EPISODE_KEYS = {"s", "r", "q"}
_GRIPPER_KEYS = {"o", "c"}
_INTERVENTION_KEY = "i"

_HELP_ACTOR = """HIL-SERL actor — keep this window focused

  i = take over with leader (rl_policy -> teleop)
  o = open gripper
  c = close gripper
  s = task successful (reward for 1 second of control frames)
  q = end episode
"""

_HELP_RECORD = """HIL-SERL recording — keep this window focused

  o = open gripper
  c = close gripper
  s = task successful (reward for 1 second of control frames)
  q = end episode
  r = rerecord episode
"""

_HELP_RECOVER = """Recover — keep this window focused

  Reconnect the USB camera cable.
  Press q when the camera is working again to end the episode.

  o = open gripper
  c = close gripper
"""


class SlobotSO100LeaderTeleop(Teleoperator):
    """Leader teleop that reports raw motor steps and exposes HIL-SERL teleop events."""

    config_class = SlobotSO100LeaderTeleopConfig
    name = "slobot_so100_leader"

    def __init__(self, config: SlobotSO100LeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self._feetech = None
        self._event_queue: Queue[str] = Queue()
        self._status_queue: Queue[tuple[str, str]] = Queue()
        self._root = None
        self._status_label = None
        self._episode_label = None
        self._help_label = None
        self._current_episode_id: int | None = None
        self._total_episodes: int | None = None
        self._success_frames_remaining = 0
        self._control_fps = 10
        self._gripper_command = GripperCommand.STAY
        self._gui_state = HilSerlGuiStateMachine(HilSerlGuiContext.ACTOR)
        self._recover_end_episode_requested = False

    @property
    def gui_state(self) -> HilSerlGuiStateMachine:
        return self._gui_state

    @property
    def gripper_command(self) -> GripperCommand:
        return self._gripper_command

    @property
    def action_features(self) -> dict:
        names = [f"{motor}.pos" for motor in Configuration.JOINT_NAMES]
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._feetech is not None and self._feetech.motors_bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._feetech is not None and self._feetech.motors_bus.is_calibrated

    def set_gui_context(self, context: HilSerlGuiContext) -> None:
        self._gui_state = HilSerlGuiStateMachine(context)
        self._refresh_gui()

    def enter_recover(self) -> None:
        self._gui_state.enter_recover()
        self._clear_success_frames()
        self._gripper_command = GripperCommand.STAY
        logger.warning("%s entering recover mode — reconnect USB camera, then press q", self)
        self._refresh_gui()

    def consume_recover_end_episode_request(self) -> bool:
        if not self._recover_end_episode_requested:
            return False
        self._recover_end_episode_requested = False
        return True

    def on_open(self) -> None:
        self._gripper_command = GripperCommand.OPEN
        self._status_queue.put(("Gripper: OPEN", "#0969da"))
        logger.info("%s gripper command: OPEN", self)

    def on_close(self) -> None:
        self._gripper_command = GripperCommand.CLOSE
        self._status_queue.put(("Gripper: CLOSE", "#0969da"))
        logger.info("%s gripper command: CLOSE", self)

    @check_if_not_connected
    def get_action(self) -> dict[str, int]:
        start = time.perf_counter()
        pos = self._feetech.get_pos()
        motor_names = list(self._feetech.motors_bus.motors.keys())
        action = {f"{name}.pos": pos[i] for i, name in enumerate(motor_names)}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        self._pump_gui()

        terminate_episode = False
        rerecord_episode = False

        while not self._event_queue.empty():
            key = self._event_queue.get_nowait()
            if key == _INTERVENTION_KEY:
                if self._gui_state.mode == HilSerlGuiMode.RECOVER:
                    continue
                if self._gui_state.context == HilSerlGuiContext.ACTOR:
                    self._gui_state.toggle_control()
                    self._refresh_gui()
            elif key == "o":
                self.on_open()
            elif key == "c":
                self.on_close()
            elif key == "s":
                if self._gui_state.mode == HilSerlGuiMode.RECOVER:
                    continue
                self._success_frames_remaining = self._control_fps
                logger.info(
                    "%s success key pressed — reward=1 for %d frames",
                    self,
                    self._success_frames_remaining,
                )
                self._status_queue.put(
                    (
                        f"Success — reward for {self._success_frames_remaining} frames",
                        "#1a7f37",
                    )
                )
            elif key == "r":
                if self._gui_state.context == HilSerlGuiContext.RECORD:
                    if self._success_frames_remaining > 0:
                        logger.info("%s success reward cancelled — reward=0 (rerecord)", self)
                    terminate_episode = True
                    rerecord_episode = True
                    self._success_frames_remaining = 0
                    self._gripper_command = GripperCommand.STAY
            elif key == "q":
                if self._gui_state.mode == HilSerlGuiMode.RECOVER:
                    self._recover_end_episode_requested = True
                    logger.info("%s recover end episode requested (q)", self)
                else:
                    terminate_episode = True

        if self.config.show_gui:
            self._pump_gui()

        success = self._success_frames_remaining > 0
        if self._success_frames_remaining > 0:
            self._success_frames_remaining -= 1
            if self._success_frames_remaining == 0:
                logger.info("%s success reward ended — reward=0", self)

        return {
            TeleopEvents.IS_INTERVENTION: self._gui_state.is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def pump_gui(self) -> None:
        """Process Tk events on the main thread (safe to call from actor / record loops)."""
        self._pump_gui()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._feetech = Factory.get_leader_feetech(
            port=self.config.port,
            robot_id=self.config.id,
            torque=False,
        )
        logger.info("%s connected.", self)
        if self.config.show_gui:
            self._start_event_gui()

    def disconnect(self) -> None:
        if self.config.show_gui:
            self._stop_event_gui()
        if self._feetech is not None:
            self._feetech.disconnect()
        logger.info("%s disconnected.", self)

    def set_control_fps(self, fps: int) -> None:
        self._control_fps = max(int(fps), 1)

    def _clear_success_frames(self) -> None:
        self._success_frames_remaining = 0

    def _help_text(self) -> str:
        if self._gui_state.mode == HilSerlGuiMode.RECOVER:
            return _HELP_RECOVER
        if self._gui_state.context == HilSerlGuiContext.RECORD:
            return _HELP_RECORD
        return _HELP_ACTOR

    def _refresh_gui(self) -> None:
        if not self.config.show_gui:
            return
        self._status_queue.put(self._gui_state.status())
        self._status_queue.put(("__help__", ""))
        self._pump_gui()

    def reset_gui_for_episode(self) -> None:
        self._clear_success_frames()
        self._gui_state.reset_for_episode()
        self._gripper_command = GripperCommand.STAY
        self._refresh_gui()

    def notify_episode(self, episode_id: int, total: int | None = None) -> None:
        self.reset_gui_for_episode()
        self._current_episode_id = episode_id
        self._total_episodes = total
        if not self.config.show_gui:
            return
        self._status_queue.put(("__episode__", ""))
        self._refresh_gui()

    def notify_resetting(self) -> None:
        self._clear_success_frames()
        self._set_status("Resetting — please wait…", fg="#c67600")

    def notify_recording_started(self) -> None:
        self.reset_gui_for_episode()

    def notify_recording_complete(self) -> None:
        self._set_status("Recording complete", fg="#1a7f37")

    def _set_status(self, text: str, fg: str) -> None:
        if not self.config.show_gui:
            return
        self._status_queue.put((text, fg))
        self._pump_gui()

    def _episode_text(self) -> str:
        if self._current_episode_id is None:
            return "Episode: —"
        if self._total_episodes is not None:
            return f"Episode: {self._current_episode_id + 1} / {self._total_episodes}"
        return f"Episode: {self._current_episode_id + 1}"

    def _apply_pending_status(self) -> None:
        while not self._status_queue.empty():
            text, fg = self._status_queue.get_nowait()
            if text == "__episode__":
                if self._episode_label is not None:
                    self._episode_label.config(text=self._episode_text())
            elif text == "__help__":
                if self._help_label is not None:
                    self._help_label.config(text=self._help_text())
            elif self._status_label is not None:
                self._status_label.config(text=text, fg=fg)

    def _on_key_press(self, event) -> None:
        key = event.char.lower() if event.char else ""
        if key in _EPISODE_KEYS or key in _GRIPPER_KEYS or key == _INTERVENTION_KEY:
            self._event_queue.put(key)

    def _start_event_gui(self) -> None:
        if self._root is not None:
            return

        import tkinter as tk

        root = tk.Tk()
        self._root = root
        root.title("HIL-SERL leader teleop")
        episode = tk.Label(root, text="Episode: —", font=("DejaVu Sans", 13, "bold"), fg="#0969da")
        episode.pack(fill=tk.X, padx=12, pady=(12, 0))
        self._episode_label = episode

        status = tk.Label(root, text="Waiting for first episode…", font=("DejaVu Sans", 12, "bold"))
        status.pack(fill=tk.X, padx=12, pady=(4, 0))
        self._status_label = status

        label = tk.Label(root, text=self._help_text(), justify=tk.LEFT, font=("DejaVu Sans", 11))
        label.pack(expand=True, fill=tk.BOTH, padx=12, pady=12)
        self._help_label = label

        for key in sorted(_EPISODE_KEYS | _GRIPPER_KEYS | {_INTERVENTION_KEY}):
            root.bind(f"<KeyPress-{key}>", self._on_key_press)
            root.bind(f"<KeyPress-{key.upper()}>", self._on_key_press)

        root.bind("<FocusIn>", lambda _e: root.focus_set())
        root.after(100, root.focus_force)
        self._refresh_gui()
        logger.info("%s event window ready.", self)

    def _pump_gui(self) -> None:
        if not self.config.show_gui:
            return
        root = self._root
        if root is None:
            return
        try:
            self._apply_pending_status()
            root.update_idletasks()
            root.update()
        except Exception:
            pass

    def _stop_event_gui(self) -> None:
        root = self._root
        self._root = None
        self._status_label = None
        self._episode_label = None
        self._help_label = None
        self._current_episode_id = None
        self._total_episodes = None
        self._success_frames_remaining = 0
        self._gripper_command = GripperCommand.STAY
        self._recover_end_episode_requested = False
        while not self._event_queue.empty():
            self._event_queue.get_nowait()
        while not self._status_queue.empty():
            self._status_queue.get_nowait()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
