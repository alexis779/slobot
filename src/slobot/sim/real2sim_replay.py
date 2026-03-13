from dataclasses import dataclass
import csv
from slobot.sim.recording_layout import RecordingLayout
from slobot.sim.configuration_mapping import ConfigurationMapping
from slobot.sim.recording_dataset_loader import RecordingDatasetLoader
from slobot.sim.sim_policy import SimPolicy
from slobot.teleop.recording_replayer import RecordingReplayer
from slobot.configuration import Configuration


class Real2SimReplay:
    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        self.sim_policy = SimPolicy(recording_id="recording_id")
        self.recording_replayer = RecordingReplayer(
            golf_ball_env=self.sim_policy.golf_ball_env,
            diff_threshold=Configuration.DIFF_THRESHOLD,
        )

    def replay_dataset(self, dataset_file: str, output_csv_file: str):
        recording_dataset_loader = RecordingDatasetLoader(dataset_file, output_csv_file)
        with open(output_csv_file, 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["episode_id", "motor_pos", "qpos", "link_quat"])
            for recording_layout in recording_dataset_loader.load_recording_layouts():
                configuration_mapping = self.replay_episode(recording_layout)
                writer.writerow([recording_layout.rrd_file, configuration_mapping.motor_pos, configuration_mapping.qpos, configuration_mapping.link_quat])

    def replay_episode(self, recording_layout: RecordingLayout):
        self.LOGGER.info(f"recording layout = {recording_layout}")

        pick_qpos, pick_link_quat = self.play_sim(recording_layout)
        pick_qpos = pick_qpos[0].tolist()
        pick_link_quat = pick_link_quat[0].tolist()

        pick_pos = self.replay_real_in_sim(recording_layout)
        pick_pos = [int(x) for x in pick_pos.tolist()]

        configuration_mapping = ConfigurationMapping(episode_id=recording_layout.rrd_file, qpos=pick_qpos, motor_pos=pick_pos, link_quat=pick_link_quat)
        self.LOGGER.info(f"configuration mapping = {configuration_mapping}")
        return configuration_mapping

    def play_sim(self, recording_layout: RecordingLayout):
        self.sim_policy.execute(recording_layout)
        return self.sim_policy.pick_qpos, self.sim_policy.pick_link_quat

    def replay_real_in_sim(self, recording_layout: RecordingLayout):
        self.recording_replayer.replay(recording_layout.rrd_file, pick_frame_id=recording_layout.pick_frame_id)
        return self.recording_replayer.pick_motor_pos