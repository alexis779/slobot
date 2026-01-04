from slobot.fps_gauge import FpsGauge
from slobot.configuration import Configuration

fps_gauge = FpsGauge(max_period=1, res=Configuration.UXGA)
fps_gauge.show_fps()