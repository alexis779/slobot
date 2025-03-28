from slobot.gradio_video_app import GradioVideoApp
from slobot.configuration import Configuration

gradio_app = GradioVideoApp(res=Configuration.SD, fps=24, segment_duration=10.0)
gradio_app.launch()