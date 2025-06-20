from slobot.feetech_frame import FeetechFrame
from slobot.configuration import Configuration

import rerun as rr

class Metrics():
    LOGGER = Configuration.logger(__name__)

    def __init__(self):
        rr.init("teleoperation", spawn=True)
        rr.log("qpos", rr.SeriesLines(names=self.metric_names("Follower")), static=True)
        rr.log("target_qpos", rr.SeriesLines(names=self.metric_names("Leader")), static=True)
        rr.log("velocity", rr.SeriesLines(names=self.metric_names("Velocity")), static=True)
        rr.log("force", rr.SeriesLines(names=self.metric_names("Force")), static=True)
        self.step = 0

    def metric_names(self, metric_name):
        return [
            f"{metric_name} {joint_name}"
            for joint_name in Configuration.JOINT_NAMES
        ]

    def handle_qpos(self, feetech_frame: FeetechFrame):
        Metrics.LOGGER.debug(f"Feetech frame {feetech_frame}")

        rr.set_time("step", sequence=self.step)
        rr.log("qpos", rr.Scalars(feetech_frame.qpos))
        rr.log("target_qpos", rr.Scalars(feetech_frame.target_qpos))
        rr.log("velocity", rr.Scalars(feetech_frame.velocity))
        rr.log("force", rr.Scalars(feetech_frame.force))

        self.step += 1
