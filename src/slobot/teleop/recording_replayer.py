import rerun as rr
from pandas import Timestamp
import genesis as gs
from importlib.resources import files

from slobot.metrics.rerun_metrics import RerunMetrics
from slobot.feetech_frame import FeetechFrame
from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.lerobot.episode_replayer import EpisodeReplayer

class RecordingReplayer():
    def __init__(self, **kwargs):
        self.rrd_file = kwargs['rrd_file']
        self.fps = kwargs['fps']
        self.period = 1.0 / self.fps
        self.feetech_frame: FeetechFrame = FeetechFrame()

        rerun_metrics = RerunMetrics()
        self.arm: SoArm100 = SoArm100(mjcf_path=Configuration.MJCF_CONFIG, fps=self.fps, should_start=False, step_handler=rerun_metrics)

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
        frames = self.replay_frames_throttled()

        first_frame = next(frames)
        self.arm.genesis.entity.set_dofs_position(first_frame.control_pos)
        self.arm.genesis.step()

        for frame in frames:
            self.arm.handle_qpos(frame)

    def replay_frames_throttled(self):
        last_frame_timestamp = None

        for frame in self.replay_frames():
            # Sample: skip frames that are too close together
            if last_frame_timestamp is not None:
                time_delta = (frame.timestamp - last_frame_timestamp).total_seconds()
                if time_delta < self.period:
                    continue

            last_frame_timestamp = frame.timestamp
            yield frame

    def replay_frames(self):
        with rr.server.Server(datasets={RerunMetrics.APPLICATION_ID: [self.rrd_file]}) as server:
            client = server.client()
            dataset = client.get_dataset(RerunMetrics.APPLICATION_ID)
            df = dataset.reader(index=RerunMetrics.TIME_METRIC)
            record_batches = df.collect()
            for record_batch in record_batches:
                for frame in self.replay_record_batch(record_batch):
                    yield frame

    def replay_record_batch(self, record_batch):
        rows = record_batch.shape[0]
        for row in range(rows):
            self.update_frame(record_batch, row)
            yield self.feetech_frame

    def update_frame(self, record_batch, row):
        self.feetech_frame.timestamp = self.get_timestamp(record_batch, row)
        self.feetech_frame.control_pos = self.get_control_pos(record_batch, row)
        self.feetech_frame.qpos = self.get_real_qpos(record_batch, row)

    def get_timestamp(self, record_batch, row) -> Timestamp:
        return self.get_cell_value(record_batch, row, 'log_time')

    def get_control_pos(self, record_batch, row):
        return self.get_metric(RerunMetrics.CONTROL_POS_METRIC, record_batch, row)

    def get_real_qpos(self, record_batch, row):
        return self.get_metric(RerunMetrics.REAL_QPOS_METRIC, record_batch, row)

    def get_metric(self, metric_name, record_batch, row):
        return [
            self.get_cell_scalar(record_batch, row, f"{metric_name}/{joint_name}:Scalars:scalars")
            for joint_name in Configuration.JOINT_NAMES
        ]

    def get_cell_scalar(self, record_batch, row, column):
        return self.get_cell(record_batch, row, column)[0].as_py()

    def get_cell(self, record_batch, row, column):
        return record_batch.column(column)[row]

    def get_cell_value(self, record_batch, row, column):
        return self.get_cell(record_batch, row, column).as_py()