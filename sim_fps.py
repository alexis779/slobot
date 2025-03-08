from slobot.fps_gauge import FpsGauge
from slobot.configuration import Configuration

# send this signal to python process to get a backtrace, for performance bottleneck analysis
import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)

fps_gauge = FpsGauge(max_period=1, res=Configuration.LD)
fps_gauge.show_fps()