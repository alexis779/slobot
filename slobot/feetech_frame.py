class FeetechFrame():
    def __init__(self, timestamp, qpos_current, qpos_goal, force):
        self.timestamp = timestamp
        self.qpos_current = qpos_current
        self.qpos_goal = qpos_goal
        self.torque = force