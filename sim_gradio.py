from slobot.gradio_app import GradioApp
from slobot.configuration import Configuration

gradio_app = GradioApp(max_fps=3, res=Configuration.SD)
gradio_app.launch()