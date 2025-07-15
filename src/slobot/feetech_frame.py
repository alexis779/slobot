class FeetechFrame():
    def __init__(self, timestamp, control_pos, qpos, velocity, control_force):
        self.timestamp = timestamp
        self.control_pos = control_pos
        self.qpos = qpos
        self.velocity = velocity
        self.control_force = control_force

    def __repr__(self):
        return f"FeetechFrame(timestamp={self.timestamp}, control_pos={self.control_pos}, qpos={self.qpos}, velocity={self.velocity}, control_force={self.control_force})"