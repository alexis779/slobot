from slobot.gradio_image_app import GradioImageApp
from slobot.configuration import Configuration

gradio_app = GradioImageApp(max_fps=3, res=Configuration.SD)
gradio_app.launch()