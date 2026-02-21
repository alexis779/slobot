from dataclasses import dataclass
import torch
import csv
from slobot.sim.recording_layout import PreGraspMode, RecordingLayout
from slobot.sim.sim_policy import SimPolicy
from slobot.teleop.recording_replayer import RecordingReplayer
from slobot.configuration import Configuration
from slobot.sim.golf_ball_env import GolfBallEnv


@dataclass
class ConfigurationMapping:
    qpos: torch.Tensor
    motor_pos: torch.Tensor


class Real2SimReplay:
    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        golf_ball_env = GolfBallEnv()
        self.sim_policy = SimPolicy(golf_ball_env)
        self.recording_replayer = RecordingReplayer(
            golf_ball_env=golf_ball_env,
            diff_threshold=Configuration.DIFF_THRESHOLD,
        )

    def replay_dataset(self, dataset_file: str, output_csv_file: str):
        with open(output_csv_file, 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["episode_id", "motor_pos", "qpos"])

            with open(dataset_file, 'r') as input_file:
                reader = csv.reader(input_file)
                next(reader)  # skip header row
                for row in reader:
                    recording_layout = RecordingLayout(
                        rrd_file=row[0],
                        pick_frame_id=int(row[1]),
                        pre_grasp_mode=PreGraspMode(row[2]),
                        ball_x=float(row[3]),
                        ball_y=float(row[4]),
                        cup_x=float(row[5]),
                        cup_y=float(row[6]),
                    )
                    configuration_mapping = self.replay_episode(recording_layout)
                    writer.writerow([recording_layout.rrd_file, configuration_mapping.motor_pos, configuration_mapping.qpos])

    def replay_episode(self, recording_layout: RecordingLayout):
        self.LOGGER.info(f"recording layout = {recording_layout}")

        pick_qpos = self.play_sim(recording_layout)
        pick_qpos = pick_qpos[0].tolist()

        pick_pos = self.replay_real_in_sim(recording_layout)
        pick_pos = [int(x) for x in pick_pos.tolist()]

        configuration_mapping = ConfigurationMapping(qpos=pick_qpos, motor_pos=pick_pos)
        self.LOGGER.info(f"configuration mapping = {configuration_mapping}")
        return configuration_mapping

    def play_sim(self, recording_layout: RecordingLayout):
        self.sim_policy.execute(recording_layout)
        return self.sim_policy.pick_qpos

    def replay_real_in_sim(self, recording_layout: RecordingLayout):
        self.recording_replayer.replay(recording_layout.rrd_file, pick_frame_id=recording_layout.pick_frame_id)
        return self.recording_replayer.pick_motor_pos