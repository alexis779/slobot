"""Explicit GUI modes for HIL-SERL leader teleop."""

from __future__ import annotations

from enum import Enum


class HilSerlGuiContext(Enum):
    ACTOR = "actor"
    RECORD = "record"


class HilSerlGuiMode(Enum):
    TELEOP = "teleop"
    RL_POLICY = "rl_policy"
    RECOVER = "recover"


class HilSerlGuiStateMachine:
    """teleop (record default) <-> rl_policy (actor default); recover on hardware errors."""

    def __init__(self, context: HilSerlGuiContext) -> None:
        self.context = context
        self.mode = self._default_mode()

    def _default_mode(self) -> HilSerlGuiMode:
        if self.context == HilSerlGuiContext.RECORD:
            return HilSerlGuiMode.TELEOP
        return HilSerlGuiMode.RL_POLICY

    @property
    def is_intervention(self) -> bool:
        return self.mode in (HilSerlGuiMode.TELEOP, HilSerlGuiMode.RECOVER)

    @property
    def is_recover(self) -> bool:
        return self.mode == HilSerlGuiMode.RECOVER

    def reset_for_episode(self) -> None:
        self.mode = self._default_mode()

    def enter_recover(self) -> None:
        self.mode = HilSerlGuiMode.RECOVER

    def resume_from_recover(self) -> None:
        self.mode = self._default_mode()

    def toggle_control(self) -> None:
        if self.mode == HilSerlGuiMode.RL_POLICY:
            self.mode = HilSerlGuiMode.TELEOP
        elif self.mode == HilSerlGuiMode.TELEOP:
            self.mode = HilSerlGuiMode.RL_POLICY

    def status(self) -> tuple[str, str]:
        if self.mode == HilSerlGuiMode.RECOVER:
            return (
                "Recover — reconnect USB camera cable, then press q to end episode",
                "#cf222e",
            )
        if self.mode == HilSerlGuiMode.TELEOP:
            if self.context == HilSerlGuiContext.RECORD:
                return ("Teleop (recording) — press q to end episode", "#c67600")
            return ("Teleop (human) — press i to hand control back to policy", "#c67600")
        return ("Policy (RL) — press i to take over with leader", "#0969da")
