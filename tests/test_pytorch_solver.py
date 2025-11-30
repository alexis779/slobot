import unittest
import torch

from slobot.rigid_body.configuration import rigid_body_configuration
from slobot.rigid_body.state import load_csv_rows
from slobot.rigid_body.pytorch_solver import PytorchSolver, make_torch_vector_factory

class TestPytorchSolver(unittest.TestCase):
    CSV_PATH = './tests/steps.csv'

    def setUp(self):
        self.pytorch_solver = PytorchSolver(device=torch.device("cpu"))
        self.vector_factory = make_torch_vector_factory(device=self.pytorch_solver.device)

    def assert_almost_equal_atol(self, actual, expected, atol):
        max_error = self.pytorch_solver.max_abs_error(actual, expected)
        self.assertTrue(max_error < atol, f"Max error {max_error} too large")

    def test_pytorch(self):
        # Load expected state from steps.csv file
        rows = load_csv_rows(self.vector_factory, self.CSV_PATH)
        
        max_step = len(rows)
        # Initialize max_step with the total number of rows in the csv
        rigid_body_configuration.max_step = max_step

        previous_entity_state = rows[0]
        for step in range(1, max_step):
            current_entity_state = rows[step]
            self.assert_step(previous_entity_state, current_entity_state)

            previous_entity_state = current_entity_state

    def assert_step(self, previous_entity_state, current_entity_state):
        # Set position and velocity using setters (will be swapped to previous_entity in step())
        self.pytorch_solver.set_pos(previous_entity_state.joint.pos)
        self.pytorch_solver.set_vel(previous_entity_state.joint.vel)

        # Call step (no parameters, no return values)
        self.pytorch_solver.step()

        # Get results using accessors
        vel = self.pytorch_solver.get_vel()
        pos = self.pytorch_solver.get_pos()
        link_quat = self.pytorch_solver.get_link_quat()
        link_pos = self.pytorch_solver.get_link_pos()

        # Get expected values
        expected_vel = current_entity_state.joint.vel
        expected_pos = current_entity_state.joint.pos
        expected_link_quat = current_entity_state.link.quat[1:, :]
        expected_link_pos = current_entity_state.link.pos[1:, :]

        self.assert_almost_equal_atol(vel, expected_vel, atol=1e-1)
        self.assert_almost_equal_atol(pos, expected_pos, atol=1e-3)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-1)
        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-1)


    def test_direct_kinematics(self):
        qpos = torch.tensor([-0.0123, -1.2707,  1.8747,  0.3543,  1.4381,  0.4008])
        vel = torch.zeros_like(qpos)
        self.pytorch_solver.set_pos(qpos)
        self.pytorch_solver.set_vel(vel)
        self.pytorch_solver.step()
        link_name = 'Fixed_Jaw'
        link_quat = self.pytorch_solver.get_link_quat(link_name)
        link_pos = self.pytorch_solver.get_link_pos(link_name)

        expected_link_pos = torch.tensor([-0.0029, -0.2843,  0.0968])
        expected_link_quat = torch.tensor([0.0617, 0.0360, 0.8852, 0.4596])

        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-1)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-1)