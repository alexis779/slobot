class FeetechFrame():
    def __init__(self, timestamp, target_qpos, qpos, velocity, force):
        self.timestamp = timestamp
        self.target_qpos = target_qpos
        self.qpos = qpos
        self.velocity = velocity
        self.force = force

    def __repr__(self):
        return f"FeetechFrame(timestamp={self.timestamp}, target_qpos={self.target_qpos}, qpos={self.qpos}, velocity={self.velocity}, force={self.force})"