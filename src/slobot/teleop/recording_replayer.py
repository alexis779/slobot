from functools import cached_property

import torch
import av

from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.lerobot.episode_replayer import EpisodeReplayer
from slobot.teleop.recording_loader import RecordingLoader
from slobot.lerobot.episode_replayer import InitialState
from slobot.lerobot.hold_state_detector import HoldStateDetector, HoldState
from slobot.lerobot.frame_delay_detector import FrameDelayDetector
from slobot.simulation_frame import SimulationFrame, CameraFrame
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase
from slobot.metrics.rerun_metrics import RerunMetrics

class RecordingReplayer:
    LOGGER = Configuration.logger(__name__)

    CAMERA_IDS = ["side", "link"]

    def __init__(self, golf_ball_env: GolfBallEnv, diff_threshold):
        self.diff_threshold = diff_threshold
        self.golf_ball_env = golf_ball_env

        self.feetech = Feetech(connect=False)

        self.step_id = 0
        self.fixed_jaw_translate = torch.tensor(self.golf_ball_env.arm.tcp_offset)

    def init_rerun_metrics(self, recording_id: str):
        self.worker_name = WorkerBase.WORKER_SIM
        self.rerun_metrics = RerunMetrics(operation_mode=WorkerBase.OPERATION_MODE, worker_name=self.worker_name)
        self.rerun_metrics.init_rerun(recording_id)

        for camera_id in RecordingReplayer.CAMERA_IDS:
            self.rerun_metrics.add_video_stream(self.camera_metric_name(camera_id), self.golf_ball_env.arm.genesis.fps)

    def replay(self, rrd_file: str, pick_frame_id: int):
        self.invalidate_cached_properties()
        self.recording_loader = RecordingLoader(rrd_file)

        self.pick_frame_id = pick_frame_id

        self.set_object_initial_positions()

        actions = self.recording_loader.observation_state # use follower state instead of leader state
        actions = [
            self.feetech.pos_to_qpos(pos)
            for pos in actions
        ]

        # Set initial position
        self.golf_ball_env.arm.genesis.entity.set_dofs_position(actions[0])
        self.step()

        # Replay remaining frames
        for step in range(1, len(actions)):
            control_pos = actions[step]
            self.golf_ball_env.arm.genesis.entity.control_dofs_position(control_pos)
            self.step()
            if step == self.pick_frame_id-self.golf_ball_env.arm.genesis.fps:
                #input("Press Enter to continue...")
                pass

        success = self.golf_ball_in_cup()
        self.LOGGER.info(f"Episode success: {success}")

    # set the initial positions of the ball and the cup
    def set_object_initial_positions(self):
        golf_ball_pos = [
            [self.get_ball_x(), self.get_ball_y(), Configuration.GOLF_BALL_RADIUS]
        ]
        self.golf_ball_env.golf_ball.set_pos(golf_ball_pos)
        self.LOGGER.info(f"initial ball position = {self.golf_ball_env.golf_ball.get_pos()}")

        cup_pos = [
            [self.initial_state.cup[0].item(), self.initial_state.cup[1].item(), 0]
        ]
        self.golf_ball_env.cup.set_pos(cup_pos)

    def get_ball_x(self):
        return self.initial_state.ball[0].item()
    
    def get_ball_y(self):
        return self.initial_state.ball[1].item()

    def invalidate_cached_properties(self):
        if hasattr(self, 'hold_state'):
            del self.hold_state
        if hasattr(self, 'initial_state'):
            del self.initial_state
        if hasattr(self, 'pick_motor_pos'):
            del self.pick_motor_pos
        if hasattr(self, 'place_motor_pos'):
            del self.place_motor_pos

    @cached_property
    def initial_state(self) -> InitialState:
        self.LOGGER.info(f"hold_state = {self.hold_state}")

        self.LOGGER.info(f"pick frame motor configuration={self.pick_motor_pos}")
        self.pick_tcp_pos = self.get_tcp_pos(self.pick_motor_pos)

        self.LOGGER.info(f"place frame motor configuration={self.place_motor_pos}")
        self.place_tcp_pos = self.get_tcp_pos(self.place_motor_pos)

        return InitialState(
            ball=self.pick_tcp_pos[0],
            cup=self.place_tcp_pos[0],
            ball_motor_pos=self.pick_motor_pos[0],
            cup_motor_pos=self.place_motor_pos[0],
        )

    @cached_property
    def pick_motor_pos(self):
        return self.get_motor_pos(self.hold_state.pick_frame_id)

    @cached_property
    def place_motor_pos(self):
        return self.get_motor_pos(self.hold_state.place_frame_id)

    @cached_property
    def hold_state(self) -> HoldState:
        leader_gripper = self.recording_loader.action[:, Configuration.GRIPPER_ID]
        follower_gripper = self.recording_loader.observation_state[:, Configuration.GRIPPER_ID]

        frame_delay_detector = FrameDelayDetector(fps=self.golf_ball_env.arm.genesis.fps)
        delay_frames = frame_delay_detector.detect_frame_delay(leader_gripper, follower_gripper)

        self.LOGGER.info(f"delay_frames = {delay_frames}")
        leader_gripper = leader_gripper[:-delay_frames]
        follower_gripper = follower_gripper[delay_frames:]

        hold_state_detector = HoldStateDetector(diff_threshold=self.diff_threshold)
        hold_state_detector.replay_teleop(leader_gripper, follower_gripper)
        hold_state = hold_state_detector.get_hold_state()

        if hold_state.pick_frame_id is None or hold_state.place_frame_id is None:
            raise ValueError("Hold state not found")

        if self.pick_frame_id is not None:
            self.LOGGER.info(f"Overriding pick frame id to {self.pick_frame_id}")
            hold_state.pick_frame_id = self.pick_frame_id

        return hold_state

    def get_motor_pos(self, frame_id):
        return self.recording_loader.frame_observation_state(frame_id)

    def get_tcp_pos(self, pos):
        qpos = self.feetech.pos_to_qpos(pos)
        self.golf_ball_env.arm.genesis.entity.set_dofs_position(qpos)
        return self.golf_ball_env.arm.genesis.link_translate(self.golf_ball_env.arm.genesis.link, self.fixed_jaw_translate)

    def golf_ball_in_cup(self):
        diff = self.golf_ball_env.golf_ball.get_pos() - self.golf_ball_env.cup.get_pos()
        diff = diff[0]
        diff = diff[:2]
        return torch.norm(diff) < Configuration.DISTANCE_THRESHOLD

    def step(self):
        self.golf_ball_env.arm.genesis.step()
        self.step_id += 1

    def stop(self):
        self.golf_ball_env.arm.genesis.stop()

        self.rerun_metrics.close_container()

    def handle_step(self, simulation_frame: SimulationFrame):
        pass
        '''
        self.rerun_metrics.handle_step(simulation_frame)

        self.rerun_metrics.log_sim_camera_frame(simulation_frame.side_camera_frame, "side")
        self.rerun_metrics.log_sim_camera_frame(simulation_frame.link_camera_frame, "link")
        '''

    def camera_metric_name(self, camera_id: str):
        return f"/{self.worker_name}/{camera_id}/video"