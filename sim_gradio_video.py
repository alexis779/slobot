from slobot.gradio_video_app import GradioVideoApp

gradio_app = GradioVideoApp(codec='libx264') # h264_nvenc
gradio_app.launch()