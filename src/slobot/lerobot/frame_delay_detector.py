import torch

class FrameDelayDetector:
    def __init__(self, fps: int):
        self.fps = fps

    def detect_frame_delay(self, leader_gripper: torch.Tensor, follower_gripper: torch.Tensor) -> int:
        min_error = float('inf')

        best_delay = 0

        for delay_frame in range(1, self.fps):
            # TODO, truncate the time series before pick frame
            gripper_diff = follower_gripper[delay_frame:] - leader_gripper[:-delay_frame]
            gripper_error = torch.norm(gripper_diff, p=2).sum()
            if gripper_error < min_error:
                min_error = gripper_error
                best_delay = delay_frame

        return best_delay
