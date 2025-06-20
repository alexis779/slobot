import time
import gradio as gr
from slobot.image_streams import ImageStreams
from slobot.configuration import Configuration

class GradioQposApp():

    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        self.image_streams = ImageStreams()

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                button = gr.Button()
                fps = gr.Slider(label='FPS', minimum=1, maximum=24, value=3, step=1)
            with gr.Row():
                shoulder_pan = gr.Number(label="shoulder_pan", precision=2)
                shoulder_lift = gr.Number(label="shoulder_lift", precision=2)
                elbow_flex = gr.Number(label="elbow_flex", precision=2)
                wrist_flex = gr.Number(label="wrist_flex", precision=2)
                wrist_roll = gr.Number(label="wrist_roll", precision=2)
                gripper = gr.Number(label="gripper", precision=2)

            button.click(self.sim_qpos, [fps], [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper])

        demo.launch()

    def sim_qpos(self, fps):
        res = Configuration.LD
        #sleep_period = 1.0 / fps
        for simulation_frame_paths in self.image_streams.simulation_frame_paths(res, fps, rgb=False, depth=False, segmentation=False, normal=False):
            #time.sleep(sleep_period)
            GradioQposApp.LOGGER.debug(f"Sending qpos {simulation_frame_paths.simulation_frame.qpos}")
            yield simulation_frame_paths.simulation_frame.qpos
