import gradio as gr
from slobot.video_streams import VideoStreams

class GradioVideoApp():
    def __init__(self, **kwargs):
        self.video_streams = VideoStreams()

        self.res = kwargs['res']
        self.fps = kwargs['fps']
        self.segment_duration = kwargs['segment_duration']

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                button = gr.Button()
                width = gr.Number(label='Width', value=640)
                height = gr.Number(label='Height', value=480)
                fps = gr.Slider(label='FPS', minimum=1, maximum=60, value=24, step=1)
                segment_duration = gr.Slider(label='Segment Duration', minimum=1, maximum=20, value=10, step=1)
            with gr.Row():
                rgb = gr.Video(label='RGB', streaming=True)
                depth = gr.Video(label='Depth', streaming=True)
            with gr.Row():
                segmentation = gr.Video(label='Segmentation Mask', streaming=True)
                surface = gr.Video(label='Surface Normal', streaming=True)

            button.click(self.sim_videos, [width, height, fps, segment_duration], [rgb, depth, segmentation, surface])

        demo.launch()

    def sim_videos(self, width, height, fps, segment_duration):
        res = (width, height)
        env_id = 0
        for simulation_frame_paths in self.video_streams.frame_filenames(res, fps, segment_duration):
            yield simulation_frame_paths.paths[env_id]
