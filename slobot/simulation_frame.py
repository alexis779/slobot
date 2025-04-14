class SimulationFrame():
    def __init__(self, timestamp, qpos, velocity, control_force):
        self.timestamp = timestamp
        self.qpos = qpos
        self.velocity = velocity
        self.control_force = control_force
        self.rgb = None
        self.depth = None
        self.segmentation = None
        self.normal = None

    def __repr__(self):
        return f"SimulationFrame(timestamp={self.timestamp}, qpos={self.qpos}, velocity={self.velocity}, control_force={self.control_force})"

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