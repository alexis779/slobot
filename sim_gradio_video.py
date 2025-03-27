from slobot.gradio_video_app import GradioVideoApp
from slobot.configuration import Configuration

gradio_app = GradioVideoApp(fps=24, res=Configuration.SD)
gradio_app.launch()