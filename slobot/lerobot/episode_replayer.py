from slobot.so_arm_100 import SoArm100
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import torch
import genesis as gs

from dataclasses import dataclass

@dataclass
class HoldState:
    pick_frame_id: int
    place_frame_id: int

@dataclass
class InitialState:
    ball_x: int
    ball_y: int
    cup_x: int
    cup_y: int


class EpisodeReplayer:
    SIGN = torch.tensor([-1, -1, 1, 1, -1, 1])

    fixed_jaw_t = torch.tensor([-2e-2, -7e-2, 0])
    golf_ball_radius = 5e-2 / 2 # 4.27e-2 / 2

    def __init__(self, **kwargs):
        self.repo_id = kwargs["repo_id"]
        ds_meta = LeRobotDatasetMetadata(self.repo_id)
        kwargs["fps"] = ds_meta.fps
        kwargs["should_build"] = False
        self.arm = SoArm100(**kwargs)

    def replay_episode(self, episode_index):
        dataset = LeRobotDataset(self.repo_id, episodes=[episode_index])

        from_idx = dataset.episode_data_index["from"][0].item()
        to_idx = dataset.episode_data_index["to"][0].item()
        episode_frame_count = to_idx - from_idx

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=episode_frame_count,
        )

        episode = next(iter(dataloader))

        hold_state = self.get_hold_state(episode)

        self.add_objects(episode, hold_state)

        for frame_id in range(episode_frame_count):
            self.replay_frame(episode, frame_id, hold_state)

    def add_objects(self, episode, hold_state : HoldState):
        self.arm.genesis.build()
        self.qpos_limits = self.arm.genesis.entity.get_dofs_limit()

        initial_state : InitialState = self.get_initial_state(episode, hold_state)

        self.arm.genesis.stop()
        self.arm.genesis.start()

        golf_ball = gs.morphs.Sphere(
            radius=self.golf_ball_radius,
            pos=(initial_state.ball_x, initial_state.ball_y, self.golf_ball_radius),
        )

        cup = gs.morphs.Mesh(
            file="./doc/cup.stl",
            pos=(initial_state.cup_x, initial_state.cup_y, 0),
        )

        self.arm.genesis.scene.add_entity(golf_ball,
                                          visualize_contact=True)
        self.arm.genesis.scene.add_entity(cup)

        self.arm.genesis.build()


    def replay_frame(self, episode, frame_id, hold_state : HoldState):
        robot_state = self.get_robot_state(episode, frame_id)
        self.arm.genesis.entity.control_dofs_position(robot_state)

        if frame_id == hold_state.pick_frame_id or frame_id == hold_state.place_frame_id:
            t = torch.tensor(self.fixed_jaw_t)
            color = (0, 1, 1, 0.4)
            self.arm.genesis.draw_arrow(self.arm.genesis.fixed_jaw, t, color)

        self.arm.genesis.step()
    
    def get_robot_state(self, episode, frame_id):
        robot_state = episode['observation.state'][frame_id]
        return self.degrees_to_radians(robot_state)

    def degrees_to_radians(self, degrees):
        radians = torch.deg2rad(degrees)
        radians = radians * self.SIGN.to(radians.device)
        radians = torch.clamp(radians, self.qpos_limits[0], self.qpos_limits[1])
        return radians

    def get_hold_state(self, episode) -> HoldState:
        gripper_id = 5
        follower_gripper = episode['action'][:,gripper_id].cpu()
        leader_gripper = episode['observation.state'][:,gripper_id].cpu()

        delay_frames = 3
        truncated_leader = leader_gripper[delay_frames:]
        gripper_diff = truncated_leader - follower_gripper[:-delay_frames]

        diff_threshold = 15
        above_threshold = torch.where(gripper_diff > diff_threshold, 1, 0)
        above_threshold_derivative = torch.diff(above_threshold, prepend=above_threshold[0:1])

        pick_frame_id, = torch.where(above_threshold_derivative == 1)
        place_frame_id, = torch.where(above_threshold_derivative == -1)
        pick_frame_id = pick_frame_id.item()
        place_frame_id = place_frame_id.item()

        return HoldState(pick_frame_id=pick_frame_id, place_frame_id=place_frame_id)

    def get_initial_state(self, episode, hold_state: HoldState):
        self.set_robot_state(episode, hold_state.pick_frame_id)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_t)
        print("target sphere pos", pick_link_pos)

        self.set_robot_state(episode, hold_state.place_frame_id)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_t)

        return InitialState(ball_x=pick_link_pos[0].item(), ball_y=pick_link_pos[1].item(), cup_x=place_link_pos[0].item(), cup_y=place_link_pos[1].item())

    def set_robot_state(self, episode, frame_id):
        robot_state = self.get_robot_state(episode, frame_id)
        self.arm.genesis.entity.set_qpos(robot_state)