import json
from pathlib import Path

import torch

from slobot.configuration import Configuration
from slobot.lerobot.episode_loader import EpisodeLoader
from slobot.rigid_body.pytorch_solver import PytorchSolver, make_torch_vector_factory
from slobot.rigid_body.state import OptimizerParametersState, from_dict, get_state_values, to_dict, load_attributes


class PytorchOptimizer:
    LOGGER = Configuration.logger(__name__)
    MAX_STEPS = 10
    STORE_STEPS = 100
    LAST_FRAMES_COUNT = 10
    PARAMETERS_STATE_FILENAME = "optimizer_parameters_state.json"

    def __init__(self, repo_id, mjcf_path, device: torch.device):
        self.repo_id = repo_id
        self.mjcf_path = mjcf_path
        self.device = device

        self.parameters_state_path = Path(Configuration.WORK_DIR) / self.PARAMETERS_STATE_FILENAME
        self.parameters_state_path.parent.mkdir(parents=True, exist_ok=True)

        self.episode_loader = EpisodeLoader(repo_id=self.repo_id, device=self.device)

        self.pytorch_solver = PytorchSolver(device=self.device)
        self.pytorch_solver.config.step_dt = 1 / self.episode_loader.dataset.meta.fps

        self.episode_loader.set_dofs_limit(
            [self.pytorch_solver.config_state.min_dofs_limit, self.pytorch_solver.config_state.max_dofs_limit]
        )

        self._load_optimizer_state()

    def _load_optimizer_state(self):
        self.optimizer_state = self._get_optimizer_state()

        self._requires_grad(self.optimizer_state)

        load_attributes(self.optimizer_state, self.pytorch_solver.config_state)

        self.episode_loader.set_middle_pos_offset(self.pytorch_solver.config_state.middle_pos_offset)

    def _build_optimizer_params(self):
        params = {
            "middle_pos_offset": self.pytorch_solver.config_state.middle_pos_offset,
            "min_force": self.pytorch_solver.config_state.min_force,
            "max_force": self.pytorch_solver.config_state.max_force,
            "Kp": self.pytorch_solver.config_state.Kp,
            "Kv": self.pytorch_solver.config_state.Kv,
            "armature": self.pytorch_solver.config_state.armature,
        }
        return self._to_optimizer_parameters_state(params)

    def _requires_grad(self, optimizer_state: OptimizerParametersState):
        for value in get_state_values(optimizer_state):
            value.requires_grad_(True)

    def _get_optimizer_state(self) -> OptimizerParametersState:
        if not self.parameters_state_path.exists():
            return self._build_optimizer_params()

        return self.read_optimizer_state()

    def read_optimizer_state(self) -> OptimizerParametersState:
        with self.parameters_state_path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return self._to_optimizer_parameters_state(data)

    def _to_optimizer_parameters_state(self, data) -> OptimizerParametersState:
        vector_factory = make_torch_vector_factory(device=self.device)
        return from_dict(OptimizerParametersState, data, vector_factory)

    def _write_optimizer_state(self, state: OptimizerParametersState):
        serializable = to_dict(state)
        with self.parameters_state_path.open("w", encoding="utf-8") as file_obj:
            json.dump(serializable, file_obj, indent=2)

    def minimize_sim_real_error(self, episode_id):
        self.episode_loader.load_episodes(episode_ids=[episode_id])

        optimizer = torch.optim.Adam(get_state_values(self.optimizer_state), lr=0.001)

        hold_state = self.episode_loader.hold_states[0]

        # discount 1/2 second worth of frames in case a collision occurred in the last frames
        last_frame_id = hold_state.pick_frame_id - int(self.episode_loader.dataset.meta.fps/2)

        for frame_id in range(1, last_frame_id):
            step = 0
            while True:
                optimizer.zero_grad()

                error = self.forward(frame_id)
                if error.item() < 0.1:
                    break  # stop once simulation error is sufficiently small

                error.backward(retain_graph=True)
                optimizer.step()
                step += 1

            PytorchOptimizer.LOGGER.info(f"episode_id {episode_id}, frame_id {frame_id}, error = {error}")

            self._write_optimizer_state(self.optimizer_state)

    def forward(self, last_frame_id):
        initial_frame_ids = [0]
        initial_follower_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.FOLLOWER_STATE_COLUMN, initial_frame_ids)
        initial_follower_robot_state = initial_follower_robot_states.squeeze(0)
        self.pytorch_solver.set_pos(initial_follower_robot_state)

        initial_follower_velocity = torch.zeros(self.pytorch_solver.config.dofs, requires_grad=True)
        self.pytorch_solver.set_vel(initial_follower_velocity)

        errors = torch.vstack([
            self.replay_frame(frame_id)
            for frame_id in range(last_frame_id+1)
        ])

        # Keep only the last frames
        last_errors = errors[-PytorchOptimizer.LAST_FRAMES_COUNT:] if errors.shape[0] >= PytorchOptimizer.LAST_FRAMES_COUNT else errors

        # Compute norm for each error vector
        norms = torch.norm(last_errors, p=2, dim=1)

        # Return the mean error norm
        mean_error = torch.mean(norms)
        PytorchOptimizer.LOGGER.info(f"mean_error = {mean_error}")
        PytorchOptimizer.LOGGER.info(f"last_errors = {last_errors}")
        return mean_error

    def replay_frame(self, frame_id):
        current_frame_ids = [frame_id]
        current_leader_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.LEADER_STATE_COLUMN, current_frame_ids)
        current_leader_robot_state = current_leader_robot_states.squeeze(0)

        self.pytorch_solver.control_dofs_position(current_leader_robot_state)

        self.pytorch_solver.step()

        next_sim_robot_state = self.pytorch_solver.get_pos()

        next_frame_ids = [frame_id + 1]
        next_follower_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.FOLLOWER_STATE_COLUMN, next_frame_ids)
        next_follower_robot_state = next_follower_robot_states.squeeze(0)

        error = next_sim_robot_state - next_follower_robot_state
        return error