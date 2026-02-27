from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from slobot.configuration import Configuration
from slobot.so_arm_100 import SoArm100
from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.sim.recording_dataset_loader import RecordingDatasetLoader
from slobot.teleop.recording_loader import RecordingLoader
from slobot.rigid_body.pytorch_solver import PytorchSolver


class Real2SimOptimizer:
    LOGGER = Configuration.logger(__name__)

    @staticmethod
    def random_init_position() -> torch.Tensor:
        """Return a random motor position within [MIN_POS, MAX_POS] for each dof."""
        min_pos = torch.tensor(SoArm100.MIN_POS, dtype=torch.float32)
        max_pos = torch.tensor(SoArm100.MAX_POS, dtype=torch.float32)
        return min_pos + (max_pos - min_pos) * torch.rand(SoArm100.DOFS)

    def __init__(self):
        self.golf_ball_env = GolfBallEnv()
        self.pytorch_solver = PytorchSolver(device=torch.device("cpu"))

    def optimize(self, input_csv_file: str, output_csv_file: str):
        self.motor_pos0 = Real2SimOptimizer.random_init_position()

        #self.motor_pos0 = [2004.385986328125, 2994, 1049.453125, 2070.1083984375, 2030.36474609375, 1902]
        self.motor_pos0 = torch.tensor(self.motor_pos0, dtype=torch.float32)

        self.motor_pos0 = torch.nn.Parameter(self.motor_pos0)

        recording_dataset_loader = RecordingDatasetLoader(input_csv_file=input_csv_file, output_csv_file=output_csv_file)

        recording_layouts = list(recording_dataset_loader.load_recording_layouts())

        configuration_mappings = list(recording_dataset_loader.load_configuration_mappings())

        pick_motor_pos = [
            RecordingLoader(recording_layout.rrd_file).frame_observation_state(recording_layout.pick_frame_id)
            for recording_layout in recording_layouts
        ]

        pick_golf_ball_pos = [
            torch.tensor([recording_layout.ball_x * Configuration.INCHES_TO_METERS, recording_layout.ball_y * Configuration.INCHES_TO_METERS, Configuration.GOLF_BALL_RADIUS])
            for recording_layout in recording_layouts
        ]

        pick_link_quat = [
            torch.tensor(configuration_mapping.link_quat)
            for configuration_mapping in configuration_mappings
        ]

        n = len(pick_motor_pos)
        train_ratio = 1.0
        n_train = max(1, int(train_ratio * n))
        pick_motor_pos_train = pick_motor_pos[:n_train]
        pick_golf_ball_pos_train = pick_golf_ball_pos[:n_train]
        pick_motor_pos_val = pick_motor_pos[n_train:]
        pick_golf_ball_pos_val = pick_golf_ball_pos[n_train:]
        pick_link_quat_train = pick_link_quat[:n_train]
        pick_link_quat_val = pick_link_quat[n_train:]
        self.LOGGER.info(f"Train/val split: {n_train} train, {len(pick_motor_pos_val)} val")

        max_steps = 10000
        optimizer = torch.optim.Adam([self.motor_pos0], lr=1.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=1.0
        )
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/tmp/slobot/real2sim/{run_id}"
        self.LOGGER.info(f"TensorBoard run id: {run_id}, log_dir: {log_dir}")
        best_loss = float("inf")
        best_motor_pos0 = None
        prev_best_loss = None
        with SummaryWriter(log_dir=log_dir) as writer:
            for step in range(max_steps):
                optimizer.zero_grad()
                loss = self.forward(pick_motor_pos_train, pick_golf_ball_pos_train, pick_link_quat_train)
                writer.add_scalar("error/train", loss.item(), step)
                val_error = None
                if pick_motor_pos_val:
                    with torch.no_grad():
                        val_error = self.forward(pick_motor_pos_val, pick_golf_ball_pos_val, pick_link_quat_val)
                        writer.add_scalar("error/validation", val_error.item(), step)
                loss.backward()

                if torch.isnan(self.motor_pos0.grad).any():
                    self.LOGGER.warning(
                        f"NaN in grad at step {step}, reinitializing motor_pos0 to random position"
                    )
                    with torch.no_grad():
                        self.motor_pos0.data.copy_(Real2SimOptimizer.random_init_position())
                    optimizer.state.clear()
                    scheduler.step()
                    prev_best_loss = None
                    continue

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_motor_pos0 = self.motor_pos0.detach().clone()

                optimizer.step()
                scheduler.step()

                val_str = f", val = {val_error.item():.6f}" if val_error is not None else ""
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("lr", lr, step)
                #self.LOGGER.info(f"step {step}, motor_pos0 = {self.motor_pos0}, grad = {self.motor_pos0.grad}, train loss = {loss.item():.6f}{val_str}")
                if step % 100 == 0:
                    self.LOGGER.info(
                        f"step {step}: motor_pos0 = {best_motor_pos0.tolist()}, train loss = {best_loss:.6f}"
                    )
                    if prev_best_loss is not None:
                        improvement = prev_best_loss - best_loss
                        if improvement < 1e-5:
                            self.LOGGER.info(
                                f"Resetting at step {step}: improvement {improvement:.2e} is too small, assuming a local minimum"
                            )

                            with torch.no_grad():
                                self.motor_pos0.data.copy_(Real2SimOptimizer.random_init_position())
                            optimizer.state.clear()
                            scheduler.step()
                            prev_best_loss = None
                            continue


                    prev_best_loss = best_loss

        self.LOGGER.info(f"best motor_pos0 = {best_motor_pos0.tolist()}, train loss = {best_loss:.6f}")

    def forward(
        self, motor_pos_list: list, golf_ball_pos_list: list, link_quat_list: list
    ) -> torch.Tensor:
        errors = [
            self.error(pick_motor_pos, pick_golf_ball_pos, pick_link_quat)
            for pick_motor_pos, pick_golf_ball_pos, pick_link_quat in zip(
                motor_pos_list, golf_ball_pos_list, link_quat_list
            )
        ]
        errors = torch.stack(errors)
        return torch.mean(errors)

    def error(self, pick_motor_pos, pick_golf_ball_pos, pick_link_quat) -> torch.Tensor:
        pick_qpos = self.qpos(pick_motor_pos)

        link_quat, link_pos = self.link_3dpose(pick_qpos)

        #self.LOGGER.info(f"pick_link_quat = {pick_link_quat}, link_quat = {link_quat}")

        pick_tcp_pos = self.tcp_pos(link_quat, link_pos)

        pos_diff = pick_golf_ball_pos - pick_tcp_pos
        pos_error = torch.norm(pos_diff)

        pick_quat_norm = torch.norm(pick_link_quat)
        link_quat_norm = torch.norm(link_quat)

        pick_quat_n = pick_link_quat / pick_quat_norm
        link_quat_n = link_quat / link_quat_norm
        dot = torch.sum(pick_quat_n * link_quat_n) # dot product of 2 quaternions is the geodesic angle
        dot = torch.clamp(dot, -1.0, 1.0)
        quat_error = 2 * torch.arccos(torch.abs(dot))

        #self.LOGGER.info(f"pos_error = {pos_error}, quat_error = {quat_error}")

        #self.debug_step(pick_qpos, pick_golf_ball_pos)

        return pos_error + quat_error/10

    def qpos(self, motor_pos):
        # self.motor_pos0 is the solution of the equation qpos(motor_pos) = 0
        return 2 * torch.pi / SoArm100.MODEL_RESOLUTION * (motor_pos - self.motor_pos0)

    def link_3dpose(self, qpos):
        self.pytorch_solver.set_pos(qpos)
        self.pytorch_solver.set_vel(torch.zeros_like(qpos))
        self.pytorch_solver.step()

        link_name = 'Fixed_Jaw'

        link_pos = self.pytorch_solver.get_link_pos(link_name)
        link_quat = self.pytorch_solver.get_link_quat(link_name)
        return link_quat, link_pos

    def tcp_pos(self, link_quat, link_pos):
        t = self.golf_ball_env.arm.tcp_offset

        t_world = self.pytorch_solver.transform_by_quat(t, link_quat)
        return link_pos + t_world


    def debug_step(self, pick_qpos, pick_golf_ball_pos):
        self.golf_ball_env.arm.genesis.entity.set_dofs_position(torch.tensor(pick_qpos, requires_grad=False))
        self.golf_ball_env.arm.genesis.scene.clear_debug_objects()
        self.golf_ball_env.arm.genesis.scene.draw_debug_sphere(pick_golf_ball_pos, Configuration.GOLF_BALL_RADIUS, color=(1, 0, 0))
        self.golf_ball_env.arm.draw_arrow_from_link_to_tcp()
        self.golf_ball_env.arm.genesis.step()
        input("Press Enter to continue...")