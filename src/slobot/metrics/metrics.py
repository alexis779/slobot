from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame
from slobot.configuration import Configuration

import rerun as rr

class Metrics():
    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        rr.init("teleoperation", spawn=True)
        rr.log("real.qpos", rr.SeriesLines(names=self.metric_names("Real Follower")), static=True)
        rr.log("control_pos", rr.SeriesLines(names=self.metric_names("Leader")), static=True)
        rr.log("real.velocity", rr.SeriesLines(names=self.metric_names("Real Velocity")), static=True)
        rr.log("real.control_force", rr.SeriesLines(names=self.metric_names("Real Control Force")), static=True)
        rr.log("sim.qpos", rr.SeriesLines(names=self.metric_names("Sim Follower")), static=True)
        rr.log("sim.velocity", rr.SeriesLines(names=self.metric_names("Sim Velocity")), static=True)
        rr.log("sim.control_force", rr.SeriesLines(names=self.metric_names("Sim Control Force")), static=True)
        self.step = 0

    def metric_names(self, metric_name):
        return [
            f"{metric_name} {joint_name}"
            for joint_name in Configuration.JOINT_NAMES
        ]

    def handle_qpos(self, feetech_frame: FeetechFrame):
        Metrics.LOGGER.debug(f"Feetech frame {feetech_frame}")

        rr.set_time("step", sequence=self.step)
        rr.log("real.qpos", rr.Scalars(feetech_frame.qpos))
        rr.log("control_pos", rr.Scalars(feetech_frame.control_pos))
        rr.log("real.velocity", rr.Scalars(feetech_frame.velocity))
        rr.log("real.control_force", rr.Scalars(feetech_frame.control_force))

        self.step += 1

    def handle_step(self, simulation_frame: SimulationFrame):
        Metrics.LOGGER.debug(f"Simulation frame {simulation_frame}")

        rr.set_time("step", sequence=self.step)
        rr.log("sim.qpos", rr.Scalars(simulation_frame.qpos))
        rr.log("control_pos", rr.Scalars(simulation_frame.control_pos))
        rr.log("sim.velocity", rr.Scalars(simulation_frame.velocity))
        rr.log("sim.control_force", rr.Scalars(simulation_frame.control_force))

        self.step += 1

    def add_metric(self, metric_name, metric_value):
        rr.log(metric_name, rr.Scalars(metric_value))