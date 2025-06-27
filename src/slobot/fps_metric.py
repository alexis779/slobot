class FpsMetric():
    def __init__(self, timestamp, fps):
        self.timestamp = timestamp
        self.fps = fps

    def __repr__(self):
        return f"FpsMetric({self.timestamp}, {self.fps})"