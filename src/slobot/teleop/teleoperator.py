from slobot.feetech import Feetech
from slobot.so_arm_100 import SoArm100
from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.lerobot.gripper_error_detector import PickFrameDetector
from slobot.lerobot.episode_replayer import EpisodeReplayer

import time
import genesis as gs

class Teleoperator():

    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        rerun_metrics = RerunMetrics()
        self.so_arm_100 = SoArm100(mjcf_path=Configuration.MJCF_CONFIG, should_start=False, step_handler=rerun_metrics)
        self.follower = Feetech(port=Feetech.PORT0, robot_id=Feetech.FOLLOWER_ID, qpos_handler=self.so_arm_100)
        self.leader = Feetech(port=Feetech.PORT1, robot_id=Feetech.LEADER_ID, torque=False)

        self.pick_frame_detector = PickFrameDetector(error_threshold=0.01, sustained_frames=5)
        self.so_arm_100.genesis.start()

        golf_ball_morph = gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=EpisodeReplayer.GOLF_BALL_RADIUS,
            pos=(0.25, 0, EpisodeReplayer.GOLF_BALL_RADIUS),
        )
        self.golf_ball = self.so_arm_100.genesis.scene.add_entity(
            golf_ball_morph,
            #material=gs.materials.Rigid(friction=1.0),
            visualize_contact=True,
        )
        self.so_arm_100.genesis.build(n_envs=1)

        self.pick_event_count = 0

    def teleoperate(self, fps):
        period = 1/fps

        while True:
            self.teleoperate_step(period)

    def teleoperate_step(self, period):
        start_time = time.time()
        leader_pos = self.leader.get_pos()
        self.follower.control_position(leader_pos)
        end_time = time.time()
        sleep_period = period - (end_time - start_time)
        if sleep_period > 0:
            time.sleep(sleep_period)