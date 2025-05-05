class FeetechFrame():
    def __init__(self, timestamp, qpos, velocity, control_force):
        self.timestamp = timestamp
        self.qpos = qpos
        self.velocity = velocity
        self.control_force = control_force

    def __repr__(self):
        return f"FeetechFrame(timestamp={self.timestamp}, qpos={self.qpos}, velocity={self.velocity}, control_force={self.control_force})"