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
                normal = gr.Image(label='Surface Normal')
            with gr.Row():
                shoulder_pan = gr.Number(label="shoulder_pan", precision=2)
                shoulder_lift = gr.Number(label="shoulder_lift", precision=2)
                elbow_flex = gr.Number(label="elbow_flex", precision=2)
                wrist_flex = gr.Number(label="wrist_flex", precision=2)
                wrist_roll = gr.Number(label="wrist_roll", precision=2)
                gripper = gr.Number(label="gripper", precision=2)

            button.click(self.sim_images, [width, height, fps], [rgb, depth, segmentation, normal, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper])

        demo.launch()

    def sim_images(self, width, height, fps):
        res = (width, height)
        for simulation_frame_paths in self.image_streams.frame_filenames(res, fps, rgb=True, depth=True, segmentation=True, normal=True):
            sim_image = []
            sim_image.extend(simulation_frame_paths.paths)
            sim_image.extend(simulation_frame_paths.qpos)
            yield sim_image