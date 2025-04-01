class SimulationFrame():
    def __init__(self, timestamp, qpos):
        self.timestamp = timestamp
        self.qpos = qpos
        self.rgb = None
        self.depth = None
        self.segmentation = None
        self.normal = None

    def frame(self, frame_id):
        match frame_id:
            case 0:
                return self.rgb
            case 1:
                return self.depth
            case 2:
                return self.segmentation
            case 3:
                return self.normal