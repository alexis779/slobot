import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from datetime import datetime
from slobot.image_streams import ImageStreams
from slobot.gravity_compensation import GravityCompensation
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame
from slobot.feetech_frame import FeetechFrame

class GradioDashboardApp():
    METRIC_CONFIG = {
        "qpos": {
            "title": "Joint Position",
            "unit": "rad",
            "real": True
        },
        "velocity": {
            "title": "Joint Velocity",
            "unit": "rad/sec",
            "real": True
        },
        "force": {
            "title": "Force",
            "unit": "N.m",
            "transform": True,
            "real": False,
        },
        "control_force": {
            "title": "Control Force",
            "unit": "N.m",
            "transform": True,
            "real": True
        }
    }

    MARKERS = [ ".", "x", "+", "<", "v", "^"]
    COLORS = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b"   # brown
    ]
    PLOTS_PER_ROW = 1
    PLOT_WIDTH = 19
    PLOT_HEIGHT = 8

    LOGGER = Configuration.logger(__name__)

    def launch(self):
        with gr.Blocks() as demo:
            df = self.sim_metrics()

            fig_width = self.PLOT_WIDTH / self.PLOTS_PER_ROW
            for joint_metric, joint_metric_config in self.METRIC_CONFIG.items():
                metric_title = joint_metric_config["title"]
                with gr.Tab(metric_title):
                    # Calculate number of rows needed
                    num_rows = (Configuration.DOFS + self.PLOTS_PER_ROW - 1) // self.PLOTS_PER_ROW  # Ceiling division
                    
                    for row in range(num_rows):
                        with gr.Row():
                            # Calculate start and end indices for this row
                            start_idx = row * self.PLOTS_PER_ROW
                            end_idx = min((row + 1) * self.PLOTS_PER_ROW, Configuration.DOFS)
                            
                            for joint_id in range(start_idx, end_idx):
                                fig = self.create_plot(df, joint_metric, joint_id, fig_width)
                                gr.Plot(fig)
                                if self.real and joint_metric_config['real']:
                                    fig = self.create_plot(df, joint_metric, joint_id, fig_width, "real_")
                                    gr.Plot(fig)

        demo.launch()

    def create_plot(self, df, metric_name, joint_id, fig_width, metrix_prefix=""):
        joint_name = Configuration.JOINT_NAMES[joint_id]
        metric_unit = self.METRIC_CONFIG[metric_name]["unit"]

        joint_metric_name = f"{metrix_prefix}{metric_name}_{joint_name}"

        fig, ax = plt.subplots(figsize=(fig_width, self.PLOT_HEIGHT))
        
        ax.plot(df["time"], df[joint_metric_name], color=self.COLORS[joint_id], marker=self.MARKERS[joint_id], markersize=5)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_unit)
        ax.set_title(joint_name)
        date_format = mdates.DateFormatter('%H:%M:%S') # Hour:Minute:Second
        ax.xaxis.set_major_formatter(date_format)

        ax.grid(True)
        
        return fig

    def sim_metrics(self, fps):
        df = pd.DataFrame()

        image_streams = ImageStreams()
        res = Configuration.LD
        for simulation_frame_paths in image_streams.simulation_frame_paths(res, fps, rgb=False, depth=False, segmentation=False, normal=False):
            simulation_frame = simulation_frame_paths.simulation_frame
            GradioDashboardApp.LOGGER.debug(f"Sending frame {simulation_frame}")
            self._update_history(df, simulation_frame)

        return df

    def _update_history(self, df: pd.DataFrame, simulation_frame: SimulationFrame):
        time = simulation_frame.timestamp
        df.loc[time, 'time'] = datetime.fromtimestamp(time)
        for joint_metric, joint_metric_config in self.METRIC_CONFIG.items():
            for joint_id in range(Configuration.DOFS):
                joint_name = Configuration.JOINT_NAMES[joint_id]
                metric_name = f"{joint_metric}_{joint_name}"
                metric_value = getattr(simulation_frame, joint_metric)[joint_id]

                #if 'transform' in joint_metric_config and joint_metric_config['transform']:
                #    metric_value = self.transform(metric_value)

                df.loc[time, metric_name] = metric_value

                if self.real and joint_metric_config['real']:
                    metric_name = f"real_{metric_name}"
                    feetech_frame: FeetechFrame = simulation_frame.feetech_frame
                    metric_value = getattr(feetech_frame, joint_metric)[joint_id]
                    df.loc[time, metric_name] = metric_value
    
    def transform(self, values):
        return self.signed_arctan(values)

    def signed_arctan(self, x, scale=1.0):
        return np.arctan(x * scale) * (2 / np.pi)  # Scales output to [-1, 1]