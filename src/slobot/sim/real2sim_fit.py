import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from slobot.configuration import Configuration
from slobot.sim.recording_dataset_loader import RecordingDatasetLoader


class Real2SimFit:
    LOGGER = Configuration.logger(__name__)

    def __init__(self, input_csv_file: str, output_csv_file: str):
        self.recording_dataset_loader = RecordingDatasetLoader(input_csv_file=input_csv_file, output_csv_file=output_csv_file)
        self.configuration_mappings = list(self.recording_dataset_loader.load_configuration_mappings())

    def fit(self):
        n_joints = len(self.configuration_mappings[0].motor_pos)
        df = pd.DataFrame(
            [
                {
                    "episode_id": configuration_mapping.episode_id,
                    **{f"motor_pos_{j}": configuration_mapping.motor_pos[j] for j in range(n_joints)},
                    **{f"qpos_{j}": configuration_mapping.qpos[j] for j in range(n_joints)},
                }
                for configuration_mapping in self.configuration_mappings
            ]
        )
        n_cols = 3
        n_rows = (n_joints + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = fig.get_axes()

        self.coefficients = []
        self.rmse = []
        self.r2 = []
        for joint_i in range(n_joints):
            pos = df[f"motor_pos_{joint_i}"].values
            qpos = df[f"qpos_{joint_i}"].values
            if joint_i == 5:
                pos = np.append(pos, 1902)
                qpos = np.append(qpos, -0.1748)
            a_i, b_i = np.polyfit(pos, qpos, 1)
            self.coefficients.append((a_i, b_i))

            qpos_pred = a_i * pos + b_i
            rmse_i = np.sqrt(np.mean((qpos - qpos_pred) ** 2))
            self.rmse.append(rmse_i)

            ss_res = np.sum((qpos - qpos_pred) ** 2)
            ss_tot = np.sum((qpos - np.mean(qpos)) ** 2)
            r2_i = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.r2.append(r2_i)

            axes[joint_i].scatter(pos, qpos)
            x_line = np.linspace(pos.min(), pos.max(), 100)
            rmse_deg = np.degrees(rmse_i)
            axes[joint_i].plot(x_line, a_i * x_line + b_i, "r-", label=f"(a, b) = ({a_i:.2e}, {b_i:.3g})\nR² = {r2_i:.3f}\nRMSE = {rmse_deg:.3g} deg")
            axes[joint_i].set_title(Configuration.JOINT_NAMES[joint_i])
            axes[joint_i].set_xlabel("motor_pos")
            axes[joint_i].set_ylabel("qpos (rad)")
            axes[joint_i].legend()

        for j in range(n_joints, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Motor position / joint angular position mapping")
        plt.tight_layout()

        motor_pos_zero = self.motor_pos_for_qpos(np.zeros(n_joints))
        self.LOGGER.info(f"motor_pos for qpos=0 (global fit): {[int(round(x)) for x in motor_pos_zero]}")

        table_rows = []
        for episode_id in sorted(df["episode_id"].unique()):
            episode_df = df[df["episode_id"] == episode_id]
            motor_pos_zero_ep = np.zeros(n_joints)
            for joint_i in range(n_joints):
                pos = episode_df[f"motor_pos_{joint_i}"].values
                qpos = episode_df[f"qpos_{joint_i}"].values
                a_global, b_global = self.coefficients[joint_i]
                if len(pos) >= 2:
                    a_i, b_i = np.polyfit(pos, qpos, 1)
                    motor_pos_zero_ep[joint_i] = (0 - b_i) / a_i
                else:
                    # Single point: use global slope, b_ep = qpos - a_global * pos
                    b_ep = float(qpos[0]) - a_global * float(pos[0])
                    motor_pos_zero_ep[joint_i] = (0 - b_ep) / a_global
            table_rows.append((episode_id, motor_pos_zero_ep))
            self.LOGGER.info(f"motor_pos for qpos=0 (episode {episode_id}): {[int(round(x)) for x in motor_pos_zero_ep]}")

        plt.show()

    def motor_pos_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Inverse mapping: given qpos, return motor_pos. qpos_i = a_i * pos_i + b_i => pos_i = (qpos_i - b_i) / a_i"""
        return np.array(
            [(qpos[i] - self.coefficients[i][1]) / self.coefficients[i][0] for i in range(len(qpos))]
        )