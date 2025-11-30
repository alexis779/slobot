import json
from pathlib import Path

import torch

from slobot.configuration import Configuration
from slobot.lerobot.episode_loader import EpisodeLoader
from slobot.rigid_body.pytorch_solver import PytorchSolver, make_torch_vector_factory
from slobot.rigid_body.state import OptimizerParametersState, from_dict, get_state_values, to_dict, load_attributes


class PytorchOptimizer:
    LOGGER = Configuration.logger(__name__)
    MAX_STEPS = 1000
    STORE_STEPS = 100
    PARAMETERS_STATE_FILENAME = "optimizer_parameters_state.json"

    def __init__(self, repo_id, mjcf_path, episode_id, device: torch.device):
        self.repo_id = repo_id
        self.mjcf_path = mjcf_path
        self.device = device

        self.parameters_state_path = Path(Configuration.WORK_DIR) / self.PARAMETERS_STATE_FILENAME
        self.parameters_state_path.parent.mkdir(parents=True, exist_ok=True)

        self.episode_loader = EpisodeLoader(repo_id=repo_id, device=device)
        self.episode_loader.load_episodes(episode_ids=[episode_id])

        self.pytorch_solver = PytorchSolver(device=device)
        self.pytorch_solver.config.step_dt = 1 / self.episode_loader.dataset.meta.fps

        self.episode_loader.set_dofs_limit(
            [self.pytorch_solver.config_state.min_dofs_limit, self.pytorch_solver.config_state.max_dofs_limit]
        )


    def minimize_sim_real_error(self):
        optimizer_state = self._get_optimizer_state()

        self._requires_grad(optimizer_state)

        load_attributes(optimizer_state, self.pytorch_solver.config_state)

        self.episode_loader.set_middle_pos_offset(self.pytorch_solver.config_state.middle_pos_offset)

        optimizer = torch.optim.Adam(get_state_values(optimizer_state))

        hold_state = self.episode_loader.hold_states[0]

        for step in range(PytorchOptimizer.MAX_STEPS):
            optimizer.zero_grad()

            # stop at the pick frame, assuming no collision occurs before that
            error = self.forward(hold_state.pick_frame_id)

            error.backward(retain_graph=True)
            optimizer.step()
            PytorchOptimizer.LOGGER.info(f"step {step}, error = {error}")
            if (step + 1) % PytorchOptimizer.STORE_STEPS == 0:
                self._write_optimizer_state(optimizer_state)

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

    def forward(self, frame_count):
        frame_errors = torch.vstack(
            [
                self.replay_frame(frame_id)
                for frame_id in range(frame_count)
            ]
        )

        return torch.mean(frame_errors)

    def replay_frame(self, frame_id):
        frame_ids = [frame_id]
        leader_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.LEADER_STATE_COLUMN, frame_ids)
        leader_robot_state = leader_robot_states.squeeze(0)

        if frame_id == 0:
            self.pytorch_solver.set_pos(leader_robot_state)
            self.pytorch_solver.set_vel(torch.zeros_like(leader_robot_state))
        else:
            self.pytorch_solver.control_dofs_position(leader_robot_state)

        self.pytorch_solver.step()

        sim_pos = self.pytorch_solver.get_pos()

        follower_robot_states = self.episode_loader.get_robot_states(EpisodeLoader.FOLLOWER_STATE_COLUMN, frame_ids)
        follower_robot_state = follower_robot_states.squeeze(0)

        error = sim_pos - follower_robot_state
        error = torch.norm(error, p=2)
        return error

