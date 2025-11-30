import unittest
import gstaichi as ti

from slobot.rigid_body.configuration import rigid_body_configuration
from slobot.rigid_body.state import load_csv_rows
from slobot.rigid_body.taichi_solver import TaichiSolver, make_taichi_vector_factory

class TestTaichiSolver(unittest.TestCase):

    def setUp(self):
        # Initialize Taichi solver with CPU backend for testing
        self.taichi_solver = TaichiSolver(arch=ti.cpu)
        self.vector_factory = make_taichi_vector_factory()

    def assert_almost_equal_atol(self, actual, expected, atol):
        """Assert that actual and expected are almost equal within tolerance."""
        max_error = self.taichi_solver.max_abs_error(actual, expected)
        self.assertTrue(max_error < atol, f"Max error {max_error} too large")

    def list_to_taichi_ndarray(self, data_list, dtype=ti.f64):
        """Create and populate a Taichi ndarray from a raw Python list.
        
        Args:
            data_list: Python list of numeric values
            dtype: Taichi data type (default: ti.f64)
        
        Returns:
            Taichi ndarray populated with values from the list
        """
        arr = ti.ndarray(dtype=dtype, shape=(len(data_list),))
        for i, val in enumerate(data_list):
            arr[i] = val
        return arr

    def test_taichi(self):
        """Test Taichi solver against expected results from CSV file."""
        # Load expected state from steps.csv file
        rows = load_csv_rows(self.vector_factory)
        
        max_step = len(rows)
        # Initialize max_step with the total number of rows in the csv
        rigid_body_configuration.max_step = max_step

        previous_entity_state = rows[0]
        for step in range(1, max_step):
            current_entity_state = rows[step]
            self.assert_step(previous_entity_state, current_entity_state)

            previous_entity_state = current_entity_state

    def assert_step(self, previous_entity_state, current_entity_state):
        """Assert that one step of the solver produces expected results."""
        # Set position and velocity using setters (will be swapped to previous_entity in step())
        # Data is already loaded as Taichi ndarrays
        self.taichi_solver.set_pos(previous_entity_state.joint.pos)
        self.taichi_solver.set_vel(previous_entity_state.joint.vel)

        # Call step (no parameters, no return values)
        self.taichi_solver.step()

        # Get actual values
        vel = self.taichi_solver.get_vel()
        pos = self.taichi_solver.get_pos()
        link_quat = self.taichi_solver.get_link_quat()
        link_pos = self.taichi_solver.get_link_pos()

        # Get expected values (already Taichi ndarrays)
        # Slice to exclude base link (first row) using solver's slice method
        expected_vel = current_entity_state.joint.vel
        expected_pos = current_entity_state.joint.pos
        expected_link_quat = self.taichi_solver._slice_ndarray(current_entity_state.link.quat, 1, None)
        expected_link_pos = self.taichi_solver._slice_ndarray(current_entity_state.link.pos, 1, None)

        self.assert_almost_equal_atol(vel, expected_vel, atol=1e-1)
        self.assert_almost_equal_atol(pos, expected_pos, atol=1e-3)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-1)
        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-1)

    def test_direct_kinematics(self):
        """Test direct kinematics for a specific link."""
        # Convert raw lists to Taichi ndarrays
        qpos_list = [-0.0123, -1.2707,  1.8747,  0.3543,  1.4381,  0.4008]
        vel_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        qpos = self.list_to_taichi_ndarray(qpos_list)
        vel = self.list_to_taichi_ndarray(vel_list)
        
        self.taichi_solver.set_pos(qpos)
        self.taichi_solver.set_vel(vel)
        self.taichi_solver.step()
        link_name = 'Fixed_Jaw'
        link_quat = self.taichi_solver.get_link_quat(link_name)
        link_pos = self.taichi_solver.get_link_pos(link_name)

        # Convert expected values to Taichi ndarrays for comparison
        expected_link_pos_list = [-0.0029, -0.2843,  0.0968]
        expected_link_quat_list = [0.0617, 0.0360, 0.8852, 0.4596]
        
        expected_link_pos = self.list_to_taichi_ndarray(expected_link_pos_list)
        expected_link_quat = self.list_to_taichi_ndarray(expected_link_quat_list)

        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-1)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-1)