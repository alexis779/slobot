import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from slobot.image_streams import ImageStreams
from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame

class GradioDashboardApp():
    METRIC_CONFIG = {
        "qpos": {
            "title": "Joint Position",
            "unit": "rad"
        },
        "velocity": {
            "title": "Joint Velocity",
            "unit": "rad/sec"
        },
        "control_force": {
            "title": "Control Force",
            "unit": "N.m"
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
    PLOT_HEIGHT = 4.8

    def __init__(self):
        self.logger = Configuration.logger(__name__)

    def launch(self):
        with gr.Blocks() as demo:
            fps = 24
            df = self.sim_metrics(fps)

            fig_width = self.PLOT_WIDTH / self.PLOTS_PER_ROW
            for metric_name in self.METRIC_CONFIG.keys():
                metric_title = self.METRIC_CONFIG[metric_name]["title"]
                with gr.Tab(metric_title):
                    # Calculate number of rows needed
                    num_rows = (Configuration.DOFS + self.PLOTS_PER_ROW - 1) // self.PLOTS_PER_ROW  # Ceiling division
                    
                    for row in range(num_rows):
                        with gr.Row():
                            # Calculate start and end indices for this row
                            start_idx = row * self.PLOTS_PER_ROW
                            end_idx = min((row + 1) * self.PLOTS_PER_ROW, Configuration.DOFS)
                            
                            for joint_id in range(start_idx, end_idx):
                                fig = self.create_plot(df, metric_name, joint_id, fig_width)
                                gr.Plot(fig)
        
        demo.launch()

    def create_plot(self, df, metric_name, joint_id, fig_width=6.4):
        joint_name = Configuration.JOINT_NAMES[joint_id]
        metric_unit = self.METRIC_CONFIG[metric_name]["unit"]

        joint_metric_name = f"{metric_name}_{joint_name}"

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
        df = self._create_df()

        image_streams = ImageStreams()
        res = Configuration.LD
        for simulation_frame_paths in image_streams.simulation_frame_paths(res, fps, rgb=False, depth=False, segmentation=False, normal=False):
            simulation_frame = simulation_frame_paths.simulation_frame
            self.logger.debug(f"Sending frame {simulation_frame}")
            self._update_history(df, simulation_frame)

        return df

    def _create_df(self):
        df = pd.DataFrame()
        
        # Add columns for each metric and joint
        for joint_metric in self.METRIC_CONFIG.keys():
            for joint_id in range(Configuration.DOFS):
                joint_name = Configuration.JOINT_NAMES[joint_id]
                metric_name = f"{joint_metric}_{joint_name}"
                df[metric_name] = None
                
        return df

    def _update_history(self, df: pd.DataFrame, simulation_frame: SimulationFrame):
        time = simulation_frame.timestamp
        df.loc[time, 'time'] = datetime.fromtimestamp(time)
        for joint_metric in self.METRIC_CONFIG.keys():
            for joint_id in range(Configuration.DOFS):
                joint_name = Configuration.JOINT_NAMES[joint_id]
                metric_name = f"{joint_metric}_{joint_name}"
                metric_value = getattr(simulation_frame, joint_metric)[joint_id]
                df.loc[time, metric_name] = metric_value