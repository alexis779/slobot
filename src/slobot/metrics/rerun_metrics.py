from slobot.feetech_frame import FeetechFrame
from slobot.simulation_frame import SimulationFrame
from slobot.configuration import Configuration

import rerun as rr
import uuid
import os

class RerunMetrics:
    LOGGER = Configuration.logger(__name__)
    APPLICATION_ID = "teleoperation"
    RRD_FOLDER = f"{Configuration.WORK_DIR}/{APPLICATION_ID}"

    def __init__(self):
        rrd_id = str(uuid.uuid4())
        rr.init(RerunMetrics.APPLICATION_ID, recording_id=rrd_id)

        os.makedirs(RerunMetrics.RRD_FOLDER, exist_ok=True)
        rr.save(f"{RerunMetrics.RRD_FOLDER}/{rrd_id}.rrd")

        for joint_name in Configuration.JOINT_NAMES:
            rr.log(f"real/qpos/{joint_name}", rr.SeriesLines(names=[f"Real Follower {joint_name}"]), static=True)
            rr.log(f"control_pos/{joint_name}", rr.SeriesLines(names=f"Leader {joint_name}"), static=True)
            rr.log(f"real/velocity/{joint_name}", rr.SeriesLines(names=f"Real Velocity {joint_name}"), static=True)
            rr.log(f"real/control_force/{joint_name}", rr.SeriesLines(names=f"Real Control Force {joint_name}"), static=True)
            rr.log(f"sim/qpos/{joint_name}", rr.SeriesLines(names=f"Sim Follower {joint_name}"), static=True)
            rr.log(f"sim/velocity/{joint_name}", rr.SeriesLines(names=f"Sim Velocity {joint_name}"), static=True)
            rr.log(f"sim/control_force/{joint_name}", rr.SeriesLines(names=f"Sim Control Force {joint_name}"), static=True)

    def handle_qpos(self, feetech_frame: FeetechFrame):
        RerunMetrics.LOGGER.debug(f"Feetech frame {feetech_frame}")

        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            rr.log(f"real/qpos/{joint_name}", rr.Scalars(feetech_frame.qpos[i]))
            rr.log(f"control_pos/{joint_name}", rr.Scalars(feetech_frame.control_pos[i]))
            rr.log(f"real/velocity/{joint_name}", rr.Scalars(feetech_frame.velocity[i]))
            rr.log(f"real/control_force/{joint_name}", rr.Scalars(feetech_frame.control_force[i]))

    def handle_step(self, simulation_frame: SimulationFrame):
        RerunMetrics.LOGGER.debug(f"Simulation frame {simulation_frame}")

        for i, joint_name in enumerate(Configuration.JOINT_NAMES):
            rr.log(f"sim/qpos/{joint_name}", rr.Scalars(simulation_frame.qpos[0][i]))
            if simulation_frame.control_pos is not None:
                rr.log(f"control_pos/{joint_name}", rr.Scalars(simulation_frame.control_pos[0][i]))
            rr.log(f"sim/velocity/{joint_name}", rr.Scalars(simulation_frame.velocity[0][i]))
            rr.log(f"sim/control_force/{joint_name}", rr.Scalars(simulation_frame.control_force[0][i]))

        if simulation_frame.feetech_frame is not None:
            self.handle_qpos(simulation_frame.feetech_frame)

    def add_metric(self, metric_name, joint_name, metric_value):
        rr.log(f"{metric_name}/{joint_name}", rr.Scalars(metric_value))