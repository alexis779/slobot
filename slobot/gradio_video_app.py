import gradio as gr
from slobot.video_streams import VideoStreams

class GradioVideoApp():
    def __init__(self, **kwargs):
        video_streams = VideoStreams()

        fps = kwargs['fps']
        res = kwargs['res']
        self.frame_filenames = video_streams.frame_filenames(fps, res)

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                button = gr.Button()
            with gr.Row():
                rgb = gr.Video(label='RGB')
                depth = gr.Video(label='Depth')
            with gr.Row():
                segmentation = gr.Video(label='Segmentation Mask')
                surface = gr.Video(label='Surface Normal')

            button.click(self.sim_videos, [], [rgb, depth, segmentation, surface])

        demo.launch()

    def sim_videos(self):
        return self.frame_filenames