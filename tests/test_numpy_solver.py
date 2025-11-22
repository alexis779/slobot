import unittest

from slobot.rigid_body.configuration import rigid_body_configuration
from slobot.rigid_body.entity_state import VectorType
from slobot.rigid_body.numpy_solver import NumpySolver
from tests.test_pytorch_solver import TestPytorchSolver

class TestNumpySolver(unittest.TestCase):

    def setUp(self):
        self.numpy_solver = NumpySolver()

    def assert_almost_equal_atol(self, actual, expected, atol):
        max_error = self.numpy_solver.max_abs_error(actual, expected)
        self.assertTrue(max_error < atol, f"Max error {max_error} too large")

    def test_numpy(self):
        # Load expected state from steps.csv file
        rows = TestPytorchSolver.load_csv_rows(VectorType.NUMPY_ARRAY)
        
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
        self.numpy_solver.set_pos(previous_entity_state.joint.pos)
        self.numpy_solver.set_vel(previous_entity_state.joint.vel)

        # Call step (no parameters, no return values)
        self.numpy_solver.step()

        # Get actual values
        vel = self.numpy_solver.get_vel()
        pos = self.numpy_solver.get_pos()
        link_quat = self.numpy_solver.get_link_quat()
        link_pos = self.numpy_solver.get_link_pos()

        # Get expected values
        expected_vel = current_entity_state.joint.vel
        expected_pos = current_entity_state.joint.pos
        expected_link_quat = current_entity_state.link.quat[1:, :]
        expected_link_pos = current_entity_state.link.pos[1:, :]

        self.assert_almost_equal_atol(vel, expected_vel, atol=1e-1)
        self.assert_almost_equal_atol(pos, expected_pos, atol=1e-3)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-1)
        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-1)