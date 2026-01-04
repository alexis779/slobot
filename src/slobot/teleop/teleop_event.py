from dataclasses import dataclass


@dataclass
class ActionEvent:
    """A general action event with start and end timestamps."""

    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Returns the duration of the action in seconds."""
        return self.end_time - self.start_time


@dataclass
class TeleopEvent:
    """Stores timing events for a single teleoperation cycle."""

    step: int
    teleop: ActionEvent
    leader_read: ActionEvent
    follower_control: ActionEvent
    follower_read: ActionEvent
    leader_qpos: list[float]
    follower_qpos: list[float]