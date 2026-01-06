from functools import cached_property

import genesis as gs
import torch
from importlib.resources import files

from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.lerobot.episode_replayer import EpisodeReplayer
from slobot.teleop.recording_loader import RecordingLoader
from slobot.lerobot.episode_replayer import InitialState
from slobot.lerobot.hold_state_detector import HoldStateDetector, HoldState
from slobot.lerobot.frame_delay_detector import FrameDelayDetector

class RecordingReplayer:
    def __init__(self, **kwargs):
        rrd_file = kwargs['rrd_file']
        self.fps = kwargs['fps']
        self.recording_loader = RecordingLoader(rrd_file)

        #rerun_metrics = RerunMetrics()
        self.arm: SoArm100 = SoArm100(mjcf_path=Configuration.MJCF_CONFIG, fps=self.fps, should_start=False, step_handler=None)

        self.build_scene()

    def build_scene(self):
        self.arm.genesis.start()

        golf_ball_morph = gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=EpisodeReplayer.GOLF_BALL_RADIUS,
            pos=(0.25, 0, EpisodeReplayer.GOLF_BALL_RADIUS),
        )
        self.golf_ball = self.arm.genesis.scene.add_entity(golf_ball_morph)

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup_morph = gs.morphs.Mesh(
            file=cup_filename,
            pos=(-0.25, 0, 0)
        )
        self.cup = self.arm.genesis.scene.add_entity(cup_morph)

        self.arm.genesis.build()

    def replay(self):
        actions = self.recording_loader.action

        # Set initial position
        self.arm.genesis.entity.set_dofs_position(actions[0])
        self.arm.genesis.step()

        self.set_object_initial_positions()

        # Replay remaining frames
        for step in range(1, len(actions)):
            if step == self.hold_state.pick_frame_id:
                self.arm.genesis.draw_arrow(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate, EpisodeReplayer.GOLF_BALL_RADIUS, (1, 0, 0, 0.5))
                print(f"golf ball position = {self.golf_ball.get_pos()}")
                #input("Pick frame")
            self.arm.genesis.entity.set_dofs_position(actions[step])
            self.arm.genesis.step()

    def set_object_initial_positions(self):
        # compute the initial positions of the ball and the cup
        golf_pos = [
            [self.initial_state.ball[0].item(), self.initial_state.ball[1].item(), EpisodeReplayer.GOLF_BALL_RADIUS]
        ]
        self.golf_ball.set_pos(golf_pos)

        cup_pos = [
            [self.initial_state.cup[0].item(), self.initial_state.cup[1].item(), 0]
        ]
        self.cup.set_pos(cup_pos)

    @cached_property
    def initial_state(self) -> InitialState:
        self.fixed_jaw_translate = torch.tensor(EpisodeReplayer.FIXED_JAW_TRANSLATE)

        self.set_robot_state(self.hold_state.pick_frame_id)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate)

        self.set_robot_state(self.hold_state.place_frame_id)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.fixed_jaw, self.fixed_jaw_translate)

        return InitialState(ball=pick_link_pos[0], cup=place_link_pos[0])

    @cached_property
    def hold_state(self) -> HoldState:
        leader_gripper = self.recording_loader.action[:, Configuration.GRIPPER_ID]
        follower_gripper = self.recording_loader.observation_state[:, Configuration.GRIPPER_ID]

        frame_delay_detector = FrameDelayDetector(fps=self.fps)
        delay_frames = frame_delay_detector.detect_frame_delay(leader_gripper, follower_gripper)

        leader_gripper = leader_gripper[:-delay_frames]
        follower_gripper = follower_gripper[delay_frames:]

        hold_state_detector = HoldStateDetector(diff_threshold=0.1)
        hold_state_detector.replay_teleop(leader_gripper, follower_gripper)
        hold_state = hold_state_detector.get_hold_state()

        if hold_state.pick_frame_id is None or hold_state.place_frame_id is None:
            raise ValueError("Hold state not found")

        return hold_state

    def set_robot_state(self, frame_id):
        robot_state = self.recording_loader.frame_observation_state(frame_id)
        self.arm.genesis.entity.set_dofs_position(robot_state)