import unittest
import csv
import json

from slobot.rigid_body.configuration import rigid_body_configuration
from slobot.rigid_body.numpy_solver import NumpySolver

import numpy as np

class TestNumpySolver(unittest.TestCase):
    def load_csv_rows(self):
        """Load rows from steps.csv file, returning tuples of (pod, vel) as numpy arrays."""
        csv_path = './tests/steps.csv'
        rows = []
        
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse JSON strings into float arrays
                pod = np.array(json.loads(row['pod']), dtype=float)
                vel = np.array(json.loads(row['vel']), dtype=float)
                rows.append((pod, vel))
        
        return rows

    def assert_almost_equal_atol(self, actual, expected, atol):
        max_error = self.numpy_solver.max_abs_error(actual, expected)
        self.assertTrue(max_error < atol, f"Max error {max_error} too large")

    def test_numpy(self):
        # Load expected pos and vel from steps.csv file
        rows = self.load_csv_rows()
        
        # Initialize max_step with the total number of rows in the csv
        max_step = len(rows)
        rigid_body_configuration.max_step=max_step

        self.numpy_solver = NumpySolver()

        for step in range(max_step-1):
            expected_pos0, expected_vel0 = rows[step]
            expected_pos1, expected_vel1 = rows[step + 1]
            self.assert_step(expected_pos0, expected_vel0, expected_pos1, expected_vel1)

    def assert_step(self, expected_pos0, expected_vel0, expected_pos1, expected_vel1):
        acc, vel, pos, mass, force = self.numpy_solver.step(expected_pos0, expected_vel0)

        self.assert_newton_euler(acc, vel, pos, mass, force, expected_pos1, expected_vel1)

    def assert_newton_euler(self, acc, vel, pos, mass, force, expected_pos, expected_vel):
        self.assert_almost_equal_atol(force, self.numpy_solver.matvec(mass, acc), atol=1e-5)

        self.assert_almost_equal_atol(pos, expected_pos, atol=1e-3)
        self.assert_almost_equal_atol(vel, expected_vel, atol=1e-1)
