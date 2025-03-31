from slobot.gradio_video_app import GradioVideoApp

gradio_app = GradioVideoApp(codec='h264_nvenc') # libx264
gradio_app.launch()