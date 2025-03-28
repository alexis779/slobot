import gradio as gr
from slobot.image_streams import ImageStreams

class GradioImageApp():
    def __init__(self):
        self.image_streams = ImageStreams()

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                button = gr.Button()
                width = gr.Number(label='Width', value=640)
                height = gr.Number(label='Height', value=480)
                fps = gr.Slider(label='FPS', minimum=1, maximum=10, value=3, step=1)
            with gr.Row():
                rgb = gr.Image(label='RGB')
                depth = gr.Image(label='Depth')
            with gr.Row():
                segmentation = gr.Image(label='Segmentation Mask')
                surface = gr.Image(label='Surface Normal')

            button.click(self.sim_images, [width, height, fps], [rgb, depth, segmentation, surface])

        demo.launch()

    def sim_images(self, width, height, fps):
        res = (width, height)
        for simulation_frame_paths in self.image_streams.frame_filenames(res, fps):
            yield simulation_frame_paths.paths
