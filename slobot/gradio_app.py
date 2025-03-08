import gradio as gr
from slobot.image_queue import ImageQueue
import time

class GradioApp():
    def __init__(self, **kwargs):
        image_queue = ImageQueue()

        max_fps = kwargs['max_fps']
        res = kwargs['res']
        self.simulation_frames = image_queue.simulation_frames(max_fps, res)

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                button = gr.Button()
            with gr.Row():
                rgb = gr.Image(label='RGB')
                depth = gr.Image(label='Depth')
            with gr.Row():
                segmentation = gr.Image(label='Segmentation Mask')
                surface = gr.Image(label='Surface Normal')

            button.click(self.sim_images, [], [rgb, depth, segmentation, surface])

        demo.launch()

    def sim_images(self):
        previous_timestamp = self.simulation_frames[0].timestamp
        for simultation_image in self.simulation_frames:
            current_timestamp = simultation_image.timestamp
            sleep_time = current_timestamp - previous_timestamp
            time.sleep(sleep_time)
            yield simultation_image.paths
            previous_timestamp = current_timestamp