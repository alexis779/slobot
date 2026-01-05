from dataclasses import dataclass

from slobot.simulation_frame import SimulationFrame


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
class SimEvent:
    """Stores data for a single simulation step."""

    step: int
    control_qpos: list[float]


@dataclass
class TeleopEvent:
    """Stores timing events for a single teleoperation cycle."""

    teleop: ActionEvent
    leader_read: ActionEvent
    follower_control: ActionEvent
    follower_read: ActionEvent
    sim_step: ActionEvent

    simulation_frame: SimulationFrame