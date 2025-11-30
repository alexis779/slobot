import unittest

from slobot.rigid_body.configuration import rigid_body_configuration
from slobot.rigid_body.scip_solver import ScipSolver
from slobot.rigid_body.numpy_solver import NumpySolver

import numpy as np

class TestScipSolver(unittest.TestCase):
    # Absolute Tolerance
    ATOL = 1e-5

    def assert_almost_equal_atol(self, actual, expected, atol = ATOL):
        max_error = self.numpy_solver.max_abs_error(actual, expected)
        self.assertTrue(max_error < atol, f"Max error {max_error} too large")

    def test_scip(self):
        max_step = 2 # < 5
        rigid_body_configuration.max_step=max_step

        self.numpy_solver = NumpySolver()
        self.scip_solver = ScipSolver()

        solved = self.scip_solver.solve()
        self.assertTrue(solved, "solution should be optimal")

        expected_pos = [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([-0.001916, -0.001734, 0.002671, 0.002481, -0.002418, 0.002415]),
            np.array([-0.005747, -0.005201, 0.008015, 0.007444, -0.007254, 0.007246]),
            np.array([-0.0115, -0.0104, 0.0160, 0.0149, -0.0145, 0.0145]),
            # np.array([-0.0192, -0.0173, 0.0267, 0.0248, -0.0242, 0.0242]),  # step 3
        ]

        expected_vel = [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([-0.191589, -0.173382, 0.267149, 0.248122, -0.241800, 0.241527]),
            np.array([-0.383153, -0.346760, 0.534308, 0.496245, -0.483598, 0.483054]),
            # np.array([-0.574657, -0.520133, 0.801490, 0.744373, -0.725393, 0.724580]),  # step 3
        ]

        for step in range(max_step-1):
            self.assert_step(step, expected_pos[step], expected_vel[step], expected_pos[step + 1])

    def assert_step(self, step: int, expected_pos0, expected_vel0, expected_pos1):
        expected_angular_jacobian, expected_linear_jacobian, expected_link_quat, expected_link_pos, expected_COM = self.assert_forward_kinematics(step, expected_pos0)

        expected_force, expected_link_cinr_pos, expected_link_cinr_inertial = self.assert_forward_dynamics(step, expected_pos0, expected_vel0,
                                                      expected_linear_jacobian, expected_angular_jacobian,
                                                      expected_link_quat, expected_link_pos,
                                                      expected_COM)

        expected_mass = self.assert_mass(step, expected_link_cinr_pos, expected_link_cinr_inertial, expected_angular_jacobian, expected_linear_jacobian)

        self.assert_newton_euler(step, expected_mass, expected_force, expected_pos0, expected_vel0, expected_pos1)

    @unittest.skip
    def test_multiple(self):
        for i in range(10):
            with self.subTest(i=i):
                self.test_scip()

    def assert_forward_kinematics(self, step: int, expected_pos0):
        expected_link_quat, expected_link_pos, expected_link_quat0, expected_link_pos0 = self.assert_link_quat_pos(step, expected_pos0)

        expected_xanchor = expected_link_pos0
        xanchor = self.scip_solver.get_xanchor(step)
        self.assert_almost_equal_atol(xanchor, expected_xanchor, atol=1e-4)

        expected_xaxis = self.numpy_solver.compute_xaxis(self.numpy_solver.config.config_state.joint_axis, expected_link_quat0)
        xaxis = self.scip_solver.get_xaxis(step)
        self.assert_almost_equal_atol(xaxis, expected_xaxis, atol=1e-3)

        expected_angular_jacobian = expected_xaxis
        angular_jacobian = self.scip_solver.get_angular_jacobian(step)
        self.assert_almost_equal_atol(angular_jacobian, expected_angular_jacobian, atol=1e-3)

        expected_COM = self.numpy_solver.compute_COM(expected_link_quat, expected_link_pos)
        COM = self.scip_solver.get_COM(step)
        self.assert_almost_equal_atol(COM, expected_COM, atol=1e-4)

        expected_angular_jacobian, expected_linear_jacobian = self.numpy_solver.compute_linear_and_angular_jacobian(expected_xaxis, expected_COM, expected_xanchor)
        linear_jacobian = self.scip_solver.get_linear_jacobian(step)
        self.assert_almost_equal_atol(linear_jacobian, expected_linear_jacobian, atol=1e-4)

        return expected_angular_jacobian, expected_linear_jacobian, expected_link_quat, expected_link_pos, expected_COM

    def assert_forward_dynamics(self, step: int,
                                expected_pos0, expected_vel0,
                                expected_linear_jacobian, expected_angular_jacobian,
                                expected_link_quat, expected_link_pos,
                                expected_COM):

        expected_bias_force, expected_link_cinr_pos, expected_link_cinr_inertial = self.assert_bias_force(step, expected_vel0,
                                                        expected_linear_jacobian, expected_angular_jacobian,
                                                        expected_link_quat, expected_link_pos,
                                                        expected_COM)

        expected_applied_force = self.assert_applied_force(step, expected_pos0, expected_vel0)

        force = self.scip_solver.get_force(step)
        expected_force = self.numpy_solver.compute_force(expected_bias_force, expected_applied_force)
        self.assert_almost_equal_atol(force, expected_force, atol=1e-4)

        return expected_force, expected_link_cinr_pos, expected_link_cinr_inertial

    def assert_newton_euler(self, step: int, expected_mass, expected_force, expected_pos0, expected_vel0, expected_pos1):
        acc = self.scip_solver.get_acc(step+1)
        force = self.scip_solver.get_force(step)
        mass = self.scip_solver.get_mass(step)
        self.assert_almost_equal_atol(force, self.numpy_solver.matvec(mass, acc))

        expected_acc, expected_vel, expected_pos = self.numpy_solver.compute_newton_euler(expected_mass, expected_force, expected_pos0, expected_vel0)
        self.assert_almost_equal_atol(expected_force, self.numpy_solver.matvec(expected_mass, expected_acc))

        self.assert_almost_equal_atol(acc, expected_acc, atol=1e0)

        vel = self.scip_solver.get_vel(step+1)
        self.assert_almost_equal_atol(vel, expected_vel, atol=1e-1)

        pos = self.scip_solver.get_pos(step+1)
        self.assert_almost_equal_atol(pos, expected_pos, atol=1e-3)
        self.assert_almost_equal_atol(pos, expected_pos1, atol=1e-3)

    def assert_applied_force(self, step: int, expected_pos0, expected_vel0):
        control_force = self.scip_solver.get_control_force(step)
        expected_control_force, expected_applied_force = self.numpy_solver.compute_applied_force(expected_pos0, expected_vel0)
        self.assert_almost_equal_atol(control_force, expected_control_force, atol=1e-1)

        applied_force = self.scip_solver.get_applied_force(step)
        self.assert_almost_equal_atol(applied_force, expected_applied_force)

        return expected_applied_force

    def assert_bias_force(self, step: int,
                        expected_vel0,
                        expected_linear_jacobian, expected_angular_jacobian,
                        expected_link_quat, expected_link_pos,
                        expected_COM):
        expected_link_cinr_inertial, expected_link_cinr_pos = self.assert_link_inertia(step, expected_link_quat, expected_link_pos, expected_COM)

        expected_f2_ang, expected_f2_vel, expected_link_angular_vel, expected_link_linear_vel = self.assert_f2(step, expected_link_cinr_inertial, expected_link_cinr_pos, expected_linear_jacobian, expected_angular_jacobian, expected_vel0)

        expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc = self.assert_joint_jacobian_acc(step, expected_link_angular_vel, expected_link_linear_vel, expected_linear_jacobian, expected_angular_jacobian)

        expected_f1_vel, expected_f1_ang = self.assert_f1(step, expected_link_cinr_inertial, expected_link_cinr_pos, expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc, expected_vel0)

        expected_link_force, expected_link_torque = self.assert_link_force_torque(step, expected_f1_vel, expected_f1_ang, expected_f2_vel, expected_f2_ang)

        expected_bias_force, expected_bias_force_angular, expected_bias_force_linear = self.numpy_solver.compute_bias_force(expected_link_torque, expected_link_force, expected_angular_jacobian, expected_linear_jacobian)

        bias_force_angular = self.scip_solver.get_bias_force_angular(step)
        self.assert_almost_equal_atol(bias_force_angular, expected_bias_force_angular, atol=1e-4)

        bias_force_linear = self.scip_solver.get_bias_force_linear(step)
        self.assert_almost_equal_atol(bias_force_linear, expected_bias_force_linear, atol=1e-4)

        bias_force = self.scip_solver.get_bias_force(step)
        self.assert_almost_equal_atol(bias_force, expected_bias_force, atol=1e-4)

        return expected_bias_force, expected_link_cinr_pos, expected_link_cinr_inertial

    def assert_f1(self, step: int, expected_link_cinr_inertia, expected_link_cinr_pos, expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc, expected_vel0):
        expected_f1_vel, expected_f1_ang, expected_link_linear_acc, expected_link_angular_acc, expected_link_linear_acc_individual, expected_link_angular_acc_individual = self.numpy_solver.compute_f1(expected_link_cinr_inertia, expected_link_cinr_pos, expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc, expected_vel0)
        link_linear_acc_individual = self.scip_solver.get_link_linear_acc_individual(step)
        self.assert_almost_equal_atol(link_linear_acc_individual, expected_link_linear_acc_individual, atol=1e-2)

        link_angular_acc_individual = self.scip_solver.get_link_angular_acc_individual(step)
        self.assert_almost_equal_atol(link_angular_acc_individual, expected_link_angular_acc_individual, atol=1e-1)

        link_linear_acc = self.scip_solver.get_link_linear_acc(step)
        self.assert_almost_equal_atol(link_linear_acc, expected_link_linear_acc, atol=1e-2)

        link_angular_acc = self.scip_solver.get_link_angular_acc(step)
        self.assert_almost_equal_atol(link_angular_acc, expected_link_angular_acc, atol=1e-1)

        f1_ang = self.scip_solver.get_f1_ang(step)
        self.assert_almost_equal_atol(f1_ang, expected_f1_ang, atol=1e-4)

        f1_vel = self.scip_solver.get_f1_vel(step)
        self.assert_almost_equal_atol(f1_vel, expected_f1_vel, atol=1e-3)

        return expected_f1_vel, expected_f1_ang

    def assert_f2(self, step: int, expected_link_cinr_inertia, expected_link_cinr_pos, expected_linear_jacobian, expected_angular_jacobian, expected_vel0):
        expected_f2_ang, expected_f2_vel, expected_link_angular_vel, expected_link_linear_vel, expected_link_angular_vel_individual, expected_link_linear_vel_individual, expected_f2_vel_vel, expected_f2_ang_vel = self.numpy_solver.compute_f2(expected_link_cinr_inertia, expected_link_cinr_pos, expected_linear_jacobian, expected_angular_jacobian, expected_vel0)

        link_linear_vel_individual = self.scip_solver.get_link_linear_vel_individual(step)
        self.assert_almost_equal_atol(link_linear_vel_individual, expected_link_linear_vel_individual, atol=1e-2)

        link_angular_vel_individual = self.scip_solver.get_link_angular_vel_individual(step)
        self.assert_almost_equal_atol(link_angular_vel_individual, expected_link_angular_vel_individual, atol=1e-1)

        link_linear_vel = self.scip_solver.get_link_linear_vel(step)
        self.assert_almost_equal_atol(link_linear_vel, expected_link_linear_vel, atol=1e-2)

        f2_vel_vel = self.scip_solver.get_f2_vel_vel(step)
        self.assert_almost_equal_atol(f2_vel_vel, expected_f2_vel_vel, atol=1e-3)

        f2_ang_vel = self.scip_solver.get_f2_ang_vel(step)
        self.assert_almost_equal_atol(f2_ang_vel, expected_f2_ang_vel, atol=1e-3)

        link_angular_vel = self.scip_solver.get_link_angular_vel(step)
        self.assert_almost_equal_atol(link_angular_vel, expected_link_angular_vel, atol=1e-1)

        f2_vel = self.scip_solver.get_f2_vel(step)
        self.assert_almost_equal_atol(f2_vel, expected_f2_vel, atol=1e-3)

        f2_ang = self.scip_solver.get_f2_ang(step)
        self.assert_almost_equal_atol(f2_ang, expected_f2_ang, atol=1e-4)

        return expected_f2_ang, expected_f2_vel, expected_link_angular_vel, expected_link_linear_vel

    def assert_link_force_torque(self, step: int, expected_f1_vel, expected_f1_ang, expected_f2_vel, expected_f2_ang):
        expected_link_force, expected_link_torque = self.numpy_solver.compute_link_force_torque(expected_f1_vel, expected_f1_ang, expected_f2_vel, expected_f2_ang)
        link_force = self.scip_solver.get_link_force(step)
        self.assert_almost_equal_atol(link_force, expected_link_force, atol=1e-3)

        link_torque = self.scip_solver.get_link_torque(step)
        self.assert_almost_equal_atol(link_torque, expected_link_torque, atol=1e-4)

        return expected_link_force, expected_link_torque

    def assert_link_quat_pos(self, step: int, expected_pos0):
        expected_link_quat, expected_link_pos, expected_link_quat0, expected_link_pos0, expected_link_rotation_vector_quat = self.numpy_solver.compute_link_quat_pos(expected_pos0)

        link_rotation_vector_quat = self.scip_solver.get_link_rotation_vector_quat(step)
        self.assert_almost_equal_atol(link_rotation_vector_quat, expected_link_rotation_vector_quat, atol=1e-4)

        link_quat = self.scip_solver.get_link_quat(step)
        self.assert_almost_equal_atol(link_quat, expected_link_quat, atol=1e-3)

        link_pos = self.scip_solver.get_link_pos(step)
        self.assert_almost_equal_atol(link_pos, expected_link_pos, atol=1e-3)

        link_quat0 = self.scip_solver.get_link_quat0(step)
        self.assert_almost_equal_atol(link_quat0, expected_link_quat0, atol=1e-3)

        link_pos0 = self.scip_solver.get_link_pos0(step)
        self.assert_almost_equal_atol(link_pos0, expected_link_pos0, atol=1e-3)

        return expected_link_quat, expected_link_pos, expected_link_quat0, expected_link_pos0

    def assert_joint_jacobian_acc(self, step: int, expected_link_angular_vel, expected_link_linear_vel, expected_linear_jacobian, expected_angular_jacobian):
        expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc = self.numpy_solver.compute_joint_jacobian_acc(expected_link_angular_vel, expected_link_linear_vel, expected_linear_jacobian, expected_angular_jacobian)
        joint_linear_jacobian_acc = self.scip_solver.get_joint_linear_jacobian_acc(step)
        self.assert_almost_equal_atol(joint_linear_jacobian_acc, expected_joint_linear_jacobian_acc, atol=1e-2)

        joint_angular_jacobian_acc = self.scip_solver.get_joint_angular_jacobian_acc(step)
        self.assert_almost_equal_atol(joint_angular_jacobian_acc, expected_joint_angular_jacobian_acc, atol=1e-1)

        return expected_joint_linear_jacobian_acc, expected_joint_angular_jacobian_acc

    def assert_link_inertia(self, step: int, expected_link_quat, expected_link_pos, expected_COM):
        expected_link_cinr_inertial, expected_link_cinr_pos, expected_link_inertial_pos = self.numpy_solver.compute_link_inertia(expected_link_quat, expected_link_pos, expected_COM)

        link_inertial_pos = self.scip_solver.get_link_inertial_pos(step)
        self.assert_almost_equal_atol(link_inertial_pos, expected_link_inertial_pos, atol=1e-4)

        link_cinr_pos = self.scip_solver.get_link_cinr_pos(step)
        self.assert_almost_equal_atol(link_cinr_pos, expected_link_cinr_pos)

        link_cinr_inertial = self.scip_solver.get_link_cinr_inertial(step)
        self.assert_almost_equal_atol(link_cinr_inertial, expected_link_cinr_inertial)

        return expected_link_cinr_inertial, expected_link_cinr_pos

    def assert_mass(self, step: int, expected_link_cinr_pos, expected_link_cinr_inertial, expected_angular_jacobian, expected_linear_jacobian):
        expected_crb_pos, expected_crb_inertial, expected_crb_mass = self.numpy_solver.compute_crb(expected_link_cinr_pos, expected_link_cinr_inertial)

        crb_pos = self.scip_solver.get_crb_pos(step)
        self.assert_almost_equal_atol(crb_pos, expected_crb_pos, atol=1e-4)

        crb_inertial = self.scip_solver.get_crb_inertial(step)
        self.assert_almost_equal_atol(crb_inertial, expected_crb_inertial)

        crb_mass = self.scip_solver.get_crb_mass(step)
        self.assert_almost_equal_atol(crb_mass, expected_crb_mass)

        expected_f_ang, expected_f_vel = self.numpy_solver.compute_f_ang_vel(expected_crb_pos, expected_crb_inertial, expected_crb_mass, expected_angular_jacobian, expected_linear_jacobian)

        f_ang = self.scip_solver.get_f_ang(step)
        self.assert_almost_equal_atol(f_ang, expected_f_ang)

        f_vel = self.scip_solver.get_f_vel(step)
        self.assert_almost_equal_atol(f_vel, expected_f_vel, atol=1e-4)

        expected_mass = self.numpy_solver.compute_mass_matrix(expected_f_ang, expected_f_vel, expected_angular_jacobian, expected_linear_jacobian)
        mass = self.scip_solver.get_mass(step)
        self.assert_almost_equal_atol(mass, expected_mass)

        return expected_mass