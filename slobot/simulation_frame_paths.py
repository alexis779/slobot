class SimulationFramePaths():
    def __init__(self, simulation_frame, paths):
        self.simulation_frame = simulation_frame
        self.paths = paths

    def __repr__(self):
        return f"SimulationFramePaths(simulation_frame={self.simulation_frame}, paths={self.paths})"