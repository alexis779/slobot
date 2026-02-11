from typing import Any
import torch

from slobot.so_arm_100 import SoArm100
from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue
from slobot.teleop.asyncprocessing.workers.sim_step_worker import SimStepWorker
from slobot.teleop.asyncprocessing.workers.worker_base import WorkerBase

from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink

class MirrorKinematicsWorker(WorkerBase):
    """Worker that runs forward kinematics on SO-ARM-100 and then inverse kinematics on another robot."""

    def __init__(
        self,
        input_queue: FifoQueue,
        fps: int,
        substeps: int,
        vis_mode: str,
        width: int,
        height: int,
        mjcf_path: str,
        end_effector_link: str,
    ):
        """Initialize the transfer kinematics worker.
        
        Args:
            input_queue: The queue to read qpos messages from
            fps: Expected frames per second
            substeps: Number of substeps
            vis_mode: Visualization mode
            width: Width of the sim RGB image
            height: Height of the sim RGB image
            mjcf_path: Path to the MJCF file for the other robot
            end_effector_link: Name of the end effector link for the other robot
        """
        super().__init__(
            worker_name=WorkerBase.WORKER_KINEMATICS,
            input_queue=input_queue,
            output_queues=[],
        )
        self.fps = fps
        self.substeps = substeps
        self.vis_mode = vis_mode
        self.width = width
        self.height = height
        self.mjcf_path = mjcf_path
        self.end_effector_link = end_effector_link

    def setup(self):
        """Initialize the mirror kinematics worker."""
        super().setup()

        res = (self.width, self.height)
        self.arm = SoArm100(should_start=False, show_viewer=True, fps=self.fps, substeps=self.substeps, rgb=True, res=res, vis_mode=self.vis_mode, camera_pos=(0.5, -2, 0.5), lookat = (0.5, 0, 0))

        self.arm.genesis.start()

        arm_morph = self.arm.genesis.parse_robot_configuration(mjcf_path=self.mjcf_path)
        self.robot: RigidEntity = self.arm.genesis.scene.add_entity(
            arm_morph,
            vis_mode=self.vis_mode,
        )

        self.robot_link: RigidLink = self.robot.get_link(self.end_effector_link)


        self.arm.genesis.build()

        self.robot_pos = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        self.robot.set_pos(self.robot_pos)

        self.LOGGER.info(f"Genesis TransferKinematicsWorker started with {self.fps} FPS, {self.substeps} substeps, {self.width}x{self.height} resolution, {self.vis_mode} visualization mode, {self.mjcf_path} MJCF path, {self.end_effector_link} end effector link")

    def process(self, so_arm_100_control_qpos: list[float]) -> tuple[int, list[float]]:
        """Run a simulation step with the given control input.
        
        Args:
            msg_type: Should be MSG_QPOS
            payload: qpos payload with target joint positions
        
        Returns:
            Tuple of (MSG_QPOS_RGB, (qpos, rgb)) - the other robot qpos and RGB image
        """
        # Convert to tensor and set joint positions
        so_arm_100_control_qpos = torch.tensor([so_arm_100_control_qpos], dtype=torch.float32)
        self.arm.genesis.entity.control_dofs_position(so_arm_100_control_qpos)

        # Perform forward kinematics on the SO-ARM-100
        fixed_jaw_pos = self.arm.genesis.fixed_jaw.get_pos()

        # translate from SO-ARM-100 base to the other robot base
        robot_link_pos = self.robot_pos + fixed_jaw_pos

        # offset the robot base due to its size
        base_offset = torch.tensor([[-0.2, -0.2, 0]], dtype=torch.float32)
        robot_link_pos = robot_link_pos + base_offset

        # Perform inverse kinematics on the other robot
        robot_qpos = self.robot.inverse_kinematics(
            link=self.robot_link,
            pos=robot_link_pos,
        )

        self.robot.control_dofs_position(robot_qpos)

        # Step the simulation
        self.arm.genesis.step()

        so_arm_100_qpos = self.arm.genesis.entity.get_qpos()
        so_arm_100_qpos = so_arm_100_qpos[0].tolist()       

        rgb, _, _, _ = self.arm.genesis.side_camera.render()

        robot_qpos = self.robot.get_qpos()
        robot_qpos = robot_qpos[0].tolist()

        return FifoQueue.MSG_QPOS_QPOS_RGB, (so_arm_100_qpos, robot_qpos, rgb)

    def publish_data(self, step: int, result_payload: Any):
        so_arm_100_qpos, robot_qpos, rgb = result_payload

        self.rerun_metrics.log_qpos(step, WorkerBase.WORKER_SIM, so_arm_100_qpos)
        self.rerun_metrics.log_rgb(step, WorkerBase.WORKER_SIM, rgb)

        self.rerun_metrics.log_qpos(step, self.worker_name, robot_qpos)
