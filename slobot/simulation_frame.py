class SimulationFrame():
    def __init__(self, timestamp, rgb, depth, segmentation, surface):
        self.timestamp = timestamp
        self.rgb = rgb
        self.depth = depth
        self.segmentation = segmentation
        self.surface = surface

    def frame(self, frame_id):
        match frame_id:
            case 0:
                return self.rgb
            case 1:
                return self.depth
            case 2:
                return self.segmentation
            case 3:
                return self.surface