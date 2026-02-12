from functools import cached_property

import genesis as gs
import torch
import json
import av
from importlib.resources import files

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
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

    def __init__(self, **kwargs):
        self.fps = kwargs['fps']

        self.golf_ball_pos_str = kwargs.get('golf_ball_pos', None)

        rrd_file = kwargs['rrd_file']
        self.recording_loader = RecordingLoader(rrd_file)

        kwargs['should_start'] = False

        kwargs['rgb'] = True
        kwargs['step_handler'] = self

        self.arm: SoArm100 = SoArm100(**kwargs)
        self.build_scene()

        self.worker_name = WorkerBase.WORKER_SIM
        self.rerun_metrics = RerunMetrics(operation_mode=WorkerBase.OPERATION_MODE, worker_name=self.worker_name)
        recording_id = "episode00"
        self.rerun_metrics.init_rerun(recording_id)

        for camera_id in RecordingReplayer.CAMERA_IDS:
            self.rerun_metrics.add_video_stream(self.camera_metric_name(camera_id))

        # initialize the video streams
        self.container = av.open("/dev/null", "w", format="h264")
        self.streams = {
            camera_id: self.create_stream() for camera_id in RecordingReplayer.CAMERA_IDS
        }


    def build_scene(self):
        vis_mode = self.arm.genesis.vis_mode

        self.arm.genesis.start()

        golf_ball_radius = EpisodeReplayer.GOLF_BALL_RADIUS
        '''
        golf_ball_morph = gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=golf_ball_radius,
            pos=(0.25, 0, golf_ball_radius),
        )
        '''
        golf_ball_morph = gs.morphs.Sphere(
            radius=golf_ball_radius,
            pos=(0.25, 0, golf_ball_radius),
        )
        self.golf_ball = self.arm.genesis.scene.add_entity(
            golf_ball_morph,
            visualize_contact=False,
            vis_mode=vis_mode,
        )

        cup_filename = str(files('slobot.config') / 'assets' / 'cup.stl')
        cup_morph = gs.morphs.Mesh(
            file=cup_filename,
            pos=(-0.25, 0, 0)
        )
        self.cup = self.arm.genesis.scene.add_entity(
            cup_morph,
            visualize_contact=False,
            vis_mode=vis_mode,
        )

        self.arm.genesis.build()

    def replay(self):
        self.set_object_initial_positions()

        actions = self.recording_loader.observation_state # use follower state instead of leader state

        #torch.save(actions, "actions.pt")

        # Set initial position
        self.step_id = 0
        self.arm.genesis.entity.set_dofs_position(actions[0])
        self.step()

        # Replay remaining frames
        for step in range(1, len(actions)):
            control_pos = actions[step]

            if step == self.hold_state.pick_frame_id:
                self.debug_tcp()

            self.arm.genesis.entity.control_dofs_position(control_pos)
            self.step()

        success = self.golf_ball_in_cup()
        RecordingReplayer.LOGGER.info(f"Episode success: {success}")

        self.stop()

    # set the initial positions of the ball and the cup
    def set_object_initial_positions(self):
        self.golf_ball.set_pos([self.golf_ball_pos])

        cup_pos = [
            [self.initial_state.cup[0].item(), self.initial_state.cup[1].item(), 0]
        ]
        self.cup.set_pos(cup_pos)

    @cached_property
    def golf_ball_pos(self):
        if self.golf_ball_pos_str is not None:
             return json.loads(self.golf_ball_pos_str)
        else:
            return [self.initial_state.ball[0].item(), self.initial_state.ball[1].item(), EpisodeReplayer.GOLF_BALL_RADIUS]

    @cached_property
    def initial_state(self) -> InitialState:
        self.fixed_jaw_translate = torch.tensor(self.arm.tcp_offset())

        self.set_robot_state(self.hold_state.pick_frame_id)
        pick_link_pos = self.arm.genesis.link_translate(self.arm.genesis.link, self.fixed_jaw_translate)

        self.set_robot_state(self.hold_state.place_frame_id)
        place_link_pos = self.arm.genesis.link_translate(self.arm.genesis.link, self.fixed_jaw_translate)

        return InitialState(ball=pick_link_pos[0], cup=place_link_pos[0])

    @cached_property
    def hold_state(self) -> HoldState:
        leader_gripper = self.recording_loader.action[:, Configuration.GRIPPER_ID]
        follower_gripper = self.recording_loader.observation_state[:, Configuration.GRIPPER_ID]

        frame_delay_detector = FrameDelayDetector(fps=self.fps)
        delay_frames = frame_delay_detector.detect_frame_delay(leader_gripper, follower_gripper)

        leader_gripper = leader_gripper[:-delay_frames]
        follower_gripper = follower_gripper[delay_frames:]

        hold_state_detector = HoldStateDetector(diff_threshold=0.3)
        hold_state_detector.replay_teleop(leader_gripper, follower_gripper)
        hold_state = hold_state_detector.get_hold_state()

        if hold_state.pick_frame_id is None or hold_state.place_frame_id is None:
            raise ValueError("Hold state not found")

        return hold_state

    def set_robot_state(self, frame_id):
        robot_state = self.recording_loader.frame_observation_state(frame_id)
        self.arm.genesis.entity.set_dofs_position(robot_state)

    def golf_ball_in_cup(self):
        diff = self.golf_ball.get_pos() - self.cup.get_pos()
        diff = diff[0]
        diff = diff[:2]
        return torch.norm(diff) < EpisodeReplayer.DISTANCE_THRESHOLD

    def debug_tcp(self):
        #self.arm.genesis.draw_arrow(self.arm.genesis.link, self.fixed_jaw_translate, EpisodeReplayer.GOLF_BALL_RADIUS, (1, 0, 0, 0.5))
        tcp_pos = self.arm.genesis.link_translate(self.arm.genesis.link, self.fixed_jaw_translate)
        RecordingReplayer.LOGGER.info(f"pick frame id = {self.hold_state.pick_frame_id}")
        RecordingReplayer.LOGGER.info(f"initial golf ball position = {self.golf_ball_pos}")
        current_golf_ball_pos = self.golf_ball.get_pos()
        RecordingReplayer.LOGGER.info(f"current golf ball position = {current_golf_ball_pos}")
        RecordingReplayer.LOGGER.info(f"TCP position = {tcp_pos}") # use this position for the golf ball initial position
        pos_offset = current_golf_ball_pos[0] - torch.tensor(self.golf_ball_pos)
        RecordingReplayer.LOGGER.info(f"pos_offset = {pos_offset}")

    def step(self):
        self.arm.genesis.step()
        self.step_id += 1

    def stop(self):
        self.arm.genesis.stop()

        # flush video streams
        for camera_id in RecordingReplayer.CAMERA_IDS:
           self.flush_stream(camera_id)

        # Close the container
        self.container.close()

    def flush_stream(self, camera_id: str):
        self.rerun_metrics.encode_frame(self.camera_metric_name(camera_id), None, self.streams[camera_id])

    def handle_step(self, simulation_frame: SimulationFrame):
        self.rerun_metrics.handle_step(simulation_frame)

        self.log_sim_camera_frame(simulation_frame.side_camera_frame, "side")
        self.log_sim_camera_frame(simulation_frame.link_camera_frame, "link")

    def log_sim_camera_frame(self, camera_frame: CameraFrame, camera_id: str):
        frame = av.VideoFrame.from_ndarray(camera_frame.rgb, format="rgb24")
        self.rerun_metrics.log_frame(self.step_id, self.camera_metric_name(camera_id), frame, self.streams[camera_id])

    def camera_metric_name(self, camera_id: str):
        return f"/{self.worker_name}/{camera_id}/video"

    def create_stream(self):
        stream = self.container.add_stream("libx264", rate=self.fps)
        #stream.max_b_frames = 0
        return stream