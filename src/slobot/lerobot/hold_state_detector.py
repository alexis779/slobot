from dataclasses import dataclass
import torch

@dataclass
class HoldState:
    pick_frame_id: int
    place_frame_id: int

class HoldStateDetector:
    DIFF_THRESHOLD = 10 # the cutoff value to identify when the gripper is holding the ball and when it is releasing the ball
    SUSTAINED_FRAMES = 4 # the number of frames the gripper must be above or below the threshold to be considered holding or releasing the ball

    def __init__(self):
        self._consecutive_above_count = 0
        self._pick_frame_id = None
        self._place_frame_id = None

    def replay_teleop(self, leader_gripper: torch.Tensor, follower_gripper: torch.Tensor):
        gripper_diff = follower_gripper - leader_gripper
        for frame_id, error in enumerate(gripper_diff):
            self._add_frame_error(frame_id, error)

    def _add_frame_error(self, frame_id: int, error: float):
        if error > self.DIFF_THRESHOLD:
            self._consecutive_above_count += 1

            if self._consecutive_above_count == self.SUSTAINED_FRAMES:
                if self._pick_frame_id is None:
                    self._pick_frame_id = frame_id - self.SUSTAINED_FRAMES + 1
        else:
            if self._consecutive_above_count > self.SUSTAINED_FRAMES:
                if self._place_frame_id is None:
                    self._place_frame_id = frame_id

            self._consecutive_above_count = 0

    def get_hold_state(self):
        return HoldState(pick_frame_id=self._pick_frame_id, place_frame_id=self._place_frame_id)