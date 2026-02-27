import csv
import json
from slobot.sim.recording_layout import RecordingLayout, PreGraspMode
from slobot.sim.configuration_mapping import ConfigurationMapping

class RecordingDatasetLoader:
    def __init__(self, input_csv_file: str, output_csv_file: str):
        self.input_csv_file = input_csv_file
        self.output_csv_file = output_csv_file

    def load_recording_layouts(self):
        with open(self.input_csv_file) as input_file:
            reader = csv.reader(input_file)
            next(reader)  # skip header row
            for row in reader:
                recording_layout = RecordingLayout(
                    rrd_file=row[0],
                    pre_pick_frame_id=int(row[1]),
                    pick_frame_id=int(row[2]),
                    pre_grasp_mode=PreGraspMode(row[3]),
                    ball_x=float(row[4]),
                    ball_y=float(row[5]),
                    cup_x=float(row[6]),
                    cup_y=float(row[7]),
                )
                yield recording_layout

    def load_configuration_mappings(self):
        with open(self.output_csv_file) as input_file:
            reader = csv.reader(input_file)
            next(reader)  # skip header row
            for row in reader:
                yield ConfigurationMapping(
                    episode_id=row[0],
                    motor_pos=json.loads(row[1]),
                    qpos=json.loads(row[2]),
                    link_quat=json.loads(row[3]),
                )