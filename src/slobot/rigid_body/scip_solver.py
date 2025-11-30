from slobot.rigid_body.configuration import Configuration, rigid_body_configuration

from pyscipopt import Model, Variable, MatrixVariable, quicksum, cos, sin
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

import numpy as np

class ScipSolver:
    # Default lower bound
    LB = -1000
    QUAT0 = [1, 0, 0, 0]

    def __init__(self):
        self.config : Configuration = rigid_body_configuration

        self._init_variables()

        self.model : Model = Model()

        self._add_variables()
        self._add_constraints()
        self._add_objective()

    def solve(self) -> bool:
        self.model.setIntParam('display/verblevel', 5) # increase verbosity compared to default value of 4
        self.model.setRealParam('numerics/epsilon', 1e-11) # default value of 1e-9 causes solver to retry with a tighter tolerance
        self.model.writeProblem('model.cip', trans=False)
        self.model.optimize()
        self.model.writeProblem('model_transformed.cip', trans=True)
        status = self.model.getStatus()
        return status == 'optimal'

    def _init_variables(self):
        # Joint variables
        self.pos : list[MatrixVariable] = self._init_variable()
        self.vel : list[MatrixVariable] = self._init_variable()
        self.acc : list[MatrixVariable] = self._init_variable()
        
        # Link kinematic variables
        self.link_linear_acc : list[MatrixVariable] = self._init_variable()
        self.link_angular_acc : list[MatrixVariable] = self._init_variable()
        self.link_linear_vel : list[MatrixVariable] = self._init_variable()
        self.link_angular_vel : list[MatrixVariable] = self._init_variable()
        self.link_pos : list[MatrixVariable] = self._init_variable()
        self.link_pos0 : list[MatrixVariable] = self._init_variable()
        self.link_relative_pos : list[MatrixVariable] = self._init_variable()
        self.link_rotation_vector : list[MatrixVariable] = self._init_variable()
        self.link_rotation_vector_quat : list[MatrixVariable] = self._init_variable()
        self.link_quat : list[MatrixVariable] = self._init_variable()
        self.link_quat0 : list[MatrixVariable] = self._init_variable()
        self.link_inertial_quat : list[MatrixVariable] = self._init_variable()
        self.link_inertial_quat_rotation : list[MatrixVariable] = self._init_variable()
        
        # Link inertial variables
        self.link_inertial_pos : list[MatrixVariable] = self._init_variable()
        self.link_inertial_pos_rotated : list[MatrixVariable] = self._init_variable()
        self.link_inertial_pos_translated : list[MatrixVariable] = self._init_variable()
        self.link_ipos : list[MatrixVariable] = self._init_variable()
        self.link_cinr_pos : list[MatrixVariable] = self._init_variable()
        self.link_cinr_inertial : list[MatrixVariable] = self._init_variable()
        self.link_inertia_rotated : list[MatrixVariable] = self._init_variable()
        self.link_inertia_hhT : list[MatrixVariable] = self._init_variable()
        self.link_inertia_hhT_mass_product : list[MatrixVariable] = self._init_variable()
        self.COM : list[MatrixVariable] = self._init_variable()
        self.crb_pos : list[MatrixVariable] = self._init_variable()
        self.crb_pos_cross_vel : list[MatrixVariable] = self._init_variable()
        self.crb_pos_cross_ang : list[MatrixVariable] = self._init_variable()
        self.crb_mass : list[MatrixVariable] = self._init_variable()
        self.crb_inertial : list[MatrixVariable] = self._init_variable()
        
        # Link force and torque variables
        self.link_force : list[MatrixVariable] = self._init_variable()
        self.link_torque : list[MatrixVariable] = self._init_variable()
        self.link_force_individual : list[MatrixVariable] = self._init_variable()
        self.link_torque_individual : list[MatrixVariable] = self._init_variable()
        self.link_linear_acc_individual : list[MatrixVariable] = self._init_variable()
        self.link_angular_acc_individual : list[MatrixVariable] = self._init_variable()
        self.link_linear_vel_individual : list[MatrixVariable] = self._init_variable()
        self.link_angular_vel_individual : list[MatrixVariable] = self._init_variable()
        
        # Joint and control variables
        self.xanchor : list[MatrixVariable] = self._init_variable()
        self.xaxis : list[MatrixVariable] = self._init_variable()
        self.angular_jacobian : list[MatrixVariable] = self._init_variable()
        self.linear_jacobian : list[MatrixVariable] = self._init_variable()
        self.force : list[MatrixVariable] = self._init_variable()
        self.bias_force : list[MatrixVariable] = self._init_variable()
        self.bias_force_angular : list[MatrixVariable] = self._init_variable()
        self.bias_force_linear : list[MatrixVariable] = self._init_variable()
        self.Kp : MatrixVariable = self._init_variable_single()
        self.Kv : MatrixVariable = self._init_variable_single()
        self.min_force : MatrixVariable = self._init_variable_single()
        self.max_force : MatrixVariable = self._init_variable_single()
        self.applied_force : list[MatrixVariable] = self._init_variable()
        self.control_force : list[MatrixVariable] = self._init_variable()
        self.control_force_higher_max : list[MatrixVariable] = self._init_variable()
        self.control_force_lower_min : list[MatrixVariable] = self._init_variable()
        self.control_force_within_range : list[MatrixVariable] = self._init_variable()
        
        # Mass and offset variables
        self.mass : list[MatrixVariable] = self._init_variable()
        self.offset_pos : list[MatrixVariable] = self._init_variable()
        
        # Force calculation variables
        self.f1_ang : list[MatrixVariable] = self._init_variable()
        self.f1_vel : list[MatrixVariable] = self._init_variable()
        self.f2_ang : list[MatrixVariable] = self._init_variable()
        self.f2_vel : list[MatrixVariable] = self._init_variable()
        self.f1_ang_mat_mul : list[MatrixVariable] = self._init_variable()
        self.f1_ang_cross_prod : list[MatrixVariable] = self._init_variable()
        self.f1_vel_constant_prod : list[MatrixVariable] = self._init_variable()
        self.f1_vel_cross_prod : list[MatrixVariable] = self._init_variable()
        self.f2_ang_vel_mat_mul : list[MatrixVariable] = self._init_variable()
        self.f2_ang_vel_cross_prod : list[MatrixVariable] = self._init_variable()
        self.f2_vel_vel_constant_prod : list[MatrixVariable] = self._init_variable()
        self.f2_vel_vel_cross_prod : list[MatrixVariable] = self._init_variable()
        self.f2_ang_vel : list[MatrixVariable] = self._init_variable()
        self.f2_vel_vel : list[MatrixVariable] = self._init_variable()
        self.f2_ang_cross_prod1 : list[MatrixVariable] = self._init_variable()
        self.f2_ang_cross_prod2 : list[MatrixVariable] = self._init_variable()
        self.f_ang : list[MatrixVariable] = self._init_variable()
        self.f_vel : list[MatrixVariable] = self._init_variable()
        
        # Joint jacobian variables
        self.joint_angular_jacobian_acc : list[MatrixVariable] = self._init_variable()
        self.joint_linear_jacobian_acc : list[MatrixVariable] = self._init_variable()
        self.joint_linear_jacobian_acc_cross_ang_vel : list[MatrixVariable] = self._init_variable()
        self.joint_linear_jacobian_acc_cross_vel_ang : list[MatrixVariable] = self._init_variable()

    def _add_variables(self):
        self.Kp = self._add_var((self.config.dofs), f"Kp")
        self.Kv = self._add_var((self.config.dofs), f"Kv")
        self.min_force = self._add_var((self.config.dofs), f"min_force")
        self.max_force = self._add_var((self.config.dofs), f"max_force")

        for step in range(self.config.max_step):
            # Joint variables
            self.pos[step] = self._add_var((self.config.dofs), f"pos_step{step}")
            self.vel[step] = self._add_var((self.config.dofs), f"vel_step{step}")
            self.acc[step] = self._add_var((self.config.dofs), f"acc_step{step}")
            
            # Link kinematic variables
            self.link_linear_acc[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_linear_acc_step{step}")
            self.link_angular_acc[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_angular_acc_step{step}")
            self.link_linear_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_linear_vel_step{step}")
            self.link_angular_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_angular_vel_step{step}")
            self.link_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_pos_step{step}")
            self.link_pos0[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_pos0_step{step}")
            self.link_relative_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_relative_pos_step{step}")
            self.link_rotation_vector[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_rotation_vector_step{step}")
            self.link_quat[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_QUAT), f"link_quat_step{step}")
            self.link_quat0[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_QUAT), f"link_quat0_step{step}")
            self.link_rotation_vector_quat[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_QUAT), f"link_rotation_vector_quat_step{step}")

            self.link_inertial_quat[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_QUAT), f"link_inertial_quat_step{step}")
            self.link_inertial_quat_rotation[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"link_inertial_quat_rotation_step{step}")
            
            # Link inertial variables
            self.link_inertial_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_inertial_pos_step{step}")
            self.link_inertial_pos_rotated[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_inertial_pos_rotated_step{step}")
            self.link_inertial_pos_translated[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_inertial_pos_translated_step{step}")
            self.link_ipos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_ipos_step{step}")
            self.link_cinr_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_cinr_pos_step{step}")
            self.link_cinr_inertial[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"link_cinr_inertial_step{step}")
            self.link_inertia_rotated[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"link_inertia_rotated_step{step}")
            self.link_inertia_hhT[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"link_inertia_hhT_step{step}")
            self.link_inertia_hhT_mass_product[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"link_inertia_hhT_mass_product_step{step}")
            # Link force and torque variables
            self.link_force[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_force_step{step}")
            self.link_torque[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_torque_step{step}")
            self.link_force_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_force_individual_step{step}")
            self.link_torque_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_torque_individual_step{step}")
            self.link_linear_acc_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_linear_acc_individual_step{step}")
            self.link_angular_acc_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_angular_acc_individual_step{step}")
            self.link_linear_vel_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_linear_vel_individual_step{step}")
            self.link_angular_vel_individual[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"link_angular_vel_individual_step{step}")
            self.COM[step] = self._add_var((Configuration.NUM_DIMS_3D), f"COM_step{step}")
            # Link CRB variables
            self.crb_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"crb_pos_step{step}")
            self.crb_pos_cross_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"crb_pos_cross_vel{step}")
            self.crb_pos_cross_ang[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"crb_pos_cross_ang{step}")

            self.crb_mass[step] = self._add_var((self.config.dofs), f"crb_mass_step{step}")

            self.crb_inertial[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D, Configuration.NUM_DIMS_3D), f"crb_inertial_step{step}")
            
            # Joint and control variables
            self.xanchor[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"xanchor_step{step}")
            self.xaxis[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"xaxis_step{step}")
            self.angular_jacobian[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"angular_jacobian_step{step}")
            self.linear_jacobian[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"linear_jacobian_step{step}")
            self.force[step] = self._add_var((self.config.dofs), f"force_step{step}")
            self.bias_force[step] = self._add_var((self.config.dofs), f"bias_force_step{step}")
            self.bias_force_angular[step] = self._add_var((self.config.dofs), f"bias_force_angular_step{step}")
            self.bias_force_linear[step] = self._add_var((self.config.dofs), f"bias_force_linear_step{step}")
            self.applied_force[step] = self._add_var((self.config.dofs), f"applied_force_step{step}")
            self.control_force[step] = self._add_var((self.config.dofs), f"control_force_step{step}")
            self.control_force_higher_max[step] = self._add_binary_var((self.config.dofs), f"control_force_higher_max_step{step}")
            self.control_force_lower_min[step] = self._add_binary_var((self.config.dofs), f"control_force_lower_min_step{step}")
            self.control_force_within_range[step] = self._add_binary_var((self.config.dofs), f"control_force_within_range_step{step}")
            
            # Mass and offset variables
            self.mass[step] = self._add_var((self.config.dofs, self.config.dofs), f"mass_step{step}")
            self.offset_pos[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"offset_pos_step{step}")
            
            # Force calculation variables
            self.f1_ang[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_ang_step{step}")
            self.f1_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_vel_step{step}")
            self.f2_ang[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_step{step}")
            self.f2_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_vel_step{step}")
            self.f1_ang_mat_mul[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_ang_mat_mul_step{step}")
            self.f1_ang_cross_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_ang_cross_prod_step{step}")
            self.f1_vel_constant_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_vel_constant_prod_step{step}")
            self.f1_vel_cross_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f1_vel_cross_prod_step{step}")
            self.f2_ang_vel_mat_mul[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_vel_mat_mul_step{step}")
            self.f2_ang_vel_cross_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_vel_cross_prod_step{step}")
            self.f2_vel_vel_constant_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_vel_vel_constant_prod_step{step}")
            self.f2_vel_vel_cross_prod[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_vel_vel_cross_prod_step{step}")
            self.f2_ang_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_vel_step{step}")
            self.f2_vel_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_vel_vel_step{step}")
            self.f2_ang_cross_prod1[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_cross_prod1_step{step}")
            self.f2_ang_cross_prod2[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f2_ang_cross_prod2_step{step}")
            self.f_ang[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f_ang_step{step}")
            self.f_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"f_vel_step{step}")

            # Joint jacobian variables
            self.joint_angular_jacobian_acc[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"joint_angular_jacobian_acc_step{step}")
            self.joint_linear_jacobian_acc[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"joint_linear_jacobian_acc_step{step}")
            self.joint_linear_jacobian_acc_cross_ang_vel[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"joint_linear_jacobian_acc_cross_ang_vel_step{step}")
            self.joint_linear_jacobian_acc_cross_vel_ang[step] = self._add_var((self.config.dofs, Configuration.NUM_DIMS_3D), f"joint_linear_jacobian_acc_cross_vel_ang_step{step}")

    def _add_constraints(self):
        self._add_initial_conditions()
        self._add_parameters()

        for step in range(self.config.max_step-1):
            self._forward_dynamics(step)

            # integrate acceleration to get velocity
            self._integrate(step, self.vel, self.acc)

            # integrate velocity to get position
            self._integrate(step, self.pos, self.vel)

            self._update_cartesian_space(step)

    def _add_initial_conditions(self):
        self.model.addMatrixCons(self.pos[0] == np.zeros(self.config.dofs))
        self.model.addMatrixCons(self.vel[0] == np.zeros(self.config.dofs))
        self.model.addMatrixCons(self.acc[0] == np.zeros(self.config.dofs))

    def _add_parameters(self):
        self.model.addMatrixCons(self.min_force == np.array(self.config.config_state.min_force))
        self.model.addMatrixCons(self.max_force == np.array(self.config.config_state.max_force))
        self.model.addMatrixCons(self.Kp == np.array(self.config.config_state.Kp))
        self.model.addMatrixCons(self.Kv == np.array(self.config.config_state.Kv))

    def _add_objective(self):
        objective = quicksum(abs(self.acc[step][dof]) for step in range(self.config.max_step) for dof in range(self.config.dofs))
        set_nonlinear_objective(self.model, objective, sense="minimize")

    def _init_variable_single(self):
        '''
        Initialize a single variable to be used across all steps.
        '''
        return None

    def _init_variable(self):
        '''
        Initialize multiple variables to be used for each step.
        '''
        return [None for _ in range(self.config.max_step)]

    # Setter methods
    def set_mass(self, mass: list[list]):
        self._set_variable(self.mass, mass)

    def set_force(self, force: list[list]):
        self._set_variable(self.force, force)

    def set_bias_force(self, bias_force: list):
        self._set_variable(self.bias_force, bias_force)

    def set_bias_force_angular(self, bias_force_angular: list):
        self._set_variable(self.bias_force_angular, bias_force_angular)

    def set_bias_force_linear(self, bias_force_linear: list):
        self._set_variable(self.bias_force_linear, bias_force_linear)

    def set_applied_force(self, applied_force: list):
        self._set_variable(self.applied_force, applied_force)

    def set_angular_jacobian(self, angular_jacobian: list[list[list]]):
        self._set_variable(self.angular_jacobian, angular_jacobian)

    def set_linear_jacobian(self, linear_jacobian: list[list]):
        self._set_variable(self.linear_jacobian, linear_jacobian)

    def set_link_cinr_pos(self, link_cinr_pos: list[list[list]]):
        self._set_variable(self.link_cinr_pos, link_cinr_pos)

    def set_link_cinr_inertial(self, link_cinr_inertial: list[list[list[list]]]):
        self._set_variable(self.link_cinr_inertial, link_cinr_inertial)

    def set_link_inertial_pos(self, step: int, link_inertial_pos: list[list]):
        self.model.addMatrixCons(self.link_inertial_pos[step] == np.array(link_inertial_pos))

    def set_link_torque(self, link_torque: list[list]):
        self._set_variable(self.link_torque, link_torque)

    def set_link_force(self, link_force: list[list]):
        self._set_variable(self.link_force, link_force)

    def set_f_ang(self, f_ang: list[list]):
         self._set_variable(self.f_ang, f_ang)

    def set_f_vel(self, f_vel: list[list]):
         self._set_variable(self.f_vel, f_vel)

    def set_link_linear_acc(self, link_linear_acc: list[list[list]]):
         self._set_variable(self.link_linear_acc, link_linear_acc)

    def set_link_angular_acc(self, link_angular_acc: list[list[list]]):
         self._set_variable(self.link_angular_acc, link_angular_acc)

    def set_link_linear_vel(self, link_linear_vel: list[list[list]]):
         self._set_variable(self.link_linear_vel, link_linear_vel)

    def set_link_angular_vel(self, link_angular_vel: list[list[list]]):
         self._set_variable(self.link_angular_vel, link_angular_vel)

    def set_joint_linear_jacobian_acc(self, joint_linear_jacobian_acc: list[list]):
         self._set_variable(self.joint_linear_jacobian_acc, joint_linear_jacobian_acc)

    def set_joint_angular_jacobian_acc(self, joint_angular_jacobian_acc: list[list]):
         self._set_variable(self.joint_angular_jacobian_acc, joint_angular_jacobian_acc)

    def set_f1_ang(self, f1_ang: list[list]):
         self._set_variable(self.f1_ang, f1_ang)

    def set_f1_vel(self, f1_vel: list[list]):
         self._set_variable(self.f1_vel, f1_vel)

    def set_f2_ang(self, f2_ang: list[list]):
         self._set_variable(self.f2_ang, f2_ang)

    def set_f2_vel(self, f2_vel: list[list]):
         self._set_variable(self.f2_vel, f2_vel)

    def set_link_quat(self, link_quat: list[list]):
        self._set_variable(self.link_quat, link_quat)

    def set_link_quat0(self, link_quat0: list[list]):
        self._set_variable(self.link_quat0, link_quat0)

    def set_link_pos(self, link_pos: list[list]):
        self._set_variable(self.link_pos, link_pos)

    def set_link_rotation_vector_quat(self, link_rotation_vector_quat: list[list[list]]):
        self._set_variable(self.link_rotation_vector_quat, link_rotation_vector_quat)

    def _set_variable(self, v: MatrixVariable, value):
        for step in range(self.config.max_step-1):
            self.model.addMatrixCons(v[step] == np.array(value[step]))

    # Getter methods - returning placeholder values for testing
    def get_xanchor(self, step: int) -> list:
        return self._get_value(self.xanchor[step])

    def get_xaxis(self, step: int) -> list:
        return self._get_value(self.xaxis[step])

    def get_angular_jacobian(self, step: int) -> list:
        return self._get_value(self.angular_jacobian[step])

    def get_linear_jacobian(self, step: int) -> list:
        return self._get_value(self.linear_jacobian[step])

    def get_force(self, step: int) -> list:
        return self._get_value(self.force[step])

    def get_bias_force(self, step: int) -> list:
        return self._get_value(self.bias_force[step])

    def get_bias_force_angular(self, step: int) -> list:
        return self._get_value(self.bias_force_angular[step])

    def get_bias_force_linear(self, step: int) -> list:
        return self._get_value(self.bias_force_linear[step])

    def get_acc(self, step: int) -> list:
        return self._get_value(self.acc[step])

    def get_mass(self, step: int) -> list:
        return self._get_value(self.mass[step])

    def get_vel(self, step: int) -> list:
        return self._get_value(self.vel[step])

    def get_pos(self, step: int) -> list:
        return self._get_value(self.pos[step])

    def get_control_force(self, step: int) -> list:
        return self._get_value(self.control_force[step])

    def get_applied_force(self, step: int) -> list:
        return self._get_value(self.applied_force[step])

    def get_link_force(self, step: int) -> list:
        return self._get_value(self.link_force[step])

    def get_link_torque(self, step: int) -> list:
        return self._get_value(self.link_torque[step])

    def get_link_linear_acc(self, step: int) -> list:
        return self._get_value(self.link_linear_acc[step])

    def get_link_linear_acc_individual(self, step: int) -> list:
        return self._get_value(self.link_linear_acc_individual[step])

    def get_link_angular_acc_individual(self, step: int) -> list:
        return self._get_value(self.link_angular_acc_individual[step])

    def get_link_linear_vel_individual(self, step: int) -> list:
        return self._get_value(self.link_linear_vel_individual[step])

    def get_link_angular_vel_individual(self, step: int) -> list:
        return self._get_value(self.link_angular_vel_individual[step])

    def get_link_angular_acc(self, step: int) -> list:
        return self._get_value(self.link_angular_acc[step])

    def get_f1_ang(self, step: int) -> list:
        return self._get_value(self.f1_ang[step])

    def get_f1_vel(self, step: int) -> list:
        return self._get_value(self.f1_vel[step])

    def get_f_ang(self, step: int) -> list:
        return self._get_value(self.f_ang[step])

    def get_f_vel(self, step: int) -> list:
        return self._get_value(self.f_vel[step])

    def get_crb_pos(self, step: int) -> list:
        return self._get_value(self.crb_pos[step])

    def get_crb_inertial(self, step: int) -> list:
        return self._get_value(self.crb_inertial[step])

    def get_crb_mass(self, step: int) -> list:
        return self._get_value(self.crb_mass[step])

    def get_link_linear_vel(self, step: int) -> list:
        return self._get_value(self.link_linear_vel[step])

    def get_link_angular_vel(self, step: int) -> list:
        return self._get_value(self.link_angular_vel[step])

    def get_f2_vel(self, step: int) -> list:
        return self._get_value(self.f2_vel[step])

    def get_f2_ang(self, step: int) -> list:
        return self._get_value(self.f2_ang[step])

    def get_f2_vel_vel(self, step: int) -> list:
        return self._get_value(self.f2_vel_vel[step])

    def get_f2_ang_vel(self, step: int) -> list:
        return self._get_value(self.f2_ang_vel[step])

    def get_link_quat(self, step: int) -> list:
        return self._get_value(self.link_quat[step])

    def get_link_pos(self, step: int) -> list:
        return self._get_value(self.link_pos[step])

    def get_link_quat0(self, step: int) -> list:
        return self._get_value(self.link_quat0[step])

    def get_link_pos0(self, step: int) -> list:
        return self._get_value(self.link_pos0[step])

    def get_link_rotation_vector_quat(self, step: int) -> list:
        return self._get_value(self.link_rotation_vector_quat[step])

    def get_joint_linear_jacobian_acc(self, step: int) -> list:
        return self._get_value(self.joint_linear_jacobian_acc[step])

    def get_joint_angular_jacobian_acc(self, step: int) -> list:
        return self._get_value(self.joint_angular_jacobian_acc[step])

    def get_link_inertial_pos(self, step: int) -> list:
        return self._get_value(self.link_inertial_pos[step])

    def get_link_cinr_pos(self, step: int) -> list:
        return self._get_value(self.link_cinr_pos[step])

    def get_link_cinr_inertial(self, step: int) -> list:
        return self._get_value(self.link_cinr_inertial[step])

    def get_link_inertial_pos_rotated(self, step: int) -> list:
        return self._get_value(self.link_inertial_pos_rotated[step])

    def get_link_ipos(self, step: int) -> list:
        return self._get_value(self.link_ipos[step])

    def get_COM(self, step: int) -> list:
        return self._get_value(self.COM[step])

    def get_link_force_individual(self, step: int) -> list:
        return self._get_value(self.link_force_individual[step])

    def get_link_torque_individual(self, step: int) -> list:
        return self._get_value(self.link_torque_individual[step])

    def _get_value(self, variable: Variable) -> list:
        return self.model.getVal(variable)

    def _add_var(self, shape, name):
        return self.model.addMatrixVar(shape, vtype='C', name=name, lb=ScipSolver.LB)

    def _add_binary_var(self, shape, name):
        return self.model.addMatrixVar(shape, vtype='B', name=name)

    def _forward_dynamics(self, step: int):
        self._mass_matrix(step)

        self._bias_force(step)
        self._applied_force(step)
        self._force(step)

        self._solve_mass(step)

    def _update_cartesian_space(self, step: int):
        self._forward_kinematics(step)
        self._COM(step)
        self._forward_velocity(step)

    def _forward_kinematics(self, step: int):
        self._link_quat_pos(step)

    def _COM(self, step: int):
        self.model.addMatrixCons(self.link_ipos[step] == self.link_inertial_pos_rotated[step] + self.link_pos[step])

        link_mass = self.config.config_state.link_mass
        link_mass_sum = sum(link_mass)

        scaled_ipos0 = np.zeros(Configuration.NUM_DIMS_3D) # Center Of Mass of the base link
        for i in range(Configuration.NUM_DIMS_3D):
            scaled_ipos0[i] = link_mass[0] * self.config.config_state.link_inertial_pos[0][i]
            self.model.addCons(self.COM[step][i] == (scaled_ipos0[i] + quicksum(link_mass[1+dof] * self.link_ipos[step][dof][i] for dof in range(self.config.dofs))) / link_mass_sum)

    def _forward_velocity(self, step: int):
        self._linear_angular_jacobian(step)

    def _mass_matrix(self, step: int):
        self._f_ang_vel(step)

        # upper triangular elements
        for dof in range(self.config.dofs):
            for dof2 in range(dof, self.config.dofs):
                work = self.f_ang[step][dof] @ self.angular_jacobian[step][dof2].T + self.f_vel[step][dof] @ self.linear_jacobian[step][dof2].T
                if dof == dof2:
                    # add diagonal term
                    diagonal = self.config.config_state.armature[dof] + self.config.step_dt * self.config.config_state.Kv[dof]
                    self.model.addMatrixCons(self.mass[step][dof, dof2] == work + diagonal)
                else:
                    self.model.addMatrixCons(self.mass[step][dof, dof2] == work)

        # lower triangular elements are copied from the upper triangular elements
        for dof in range(self.config.dofs):
            for dof2 in range(dof):
                self.model.addCons(self.mass[step][dof, dof2] == self.mass[step][dof2, dof])

    def _f_ang_vel(self, step: int):
        self._crb(step)

        self._cross_product(self.crb_pos_cross_vel[step], self.crb_pos[step], self.linear_jacobian[step])
        self._cross_product(self.crb_pos_cross_ang[step], self.crb_pos[step], self.angular_jacobian[step])

        for dof in range(self.config.dofs):
            self.model.addMatrixCons(self.f_ang[step][dof] == self.crb_inertial[step][dof] @ self.angular_jacobian[step][dof] + self.crb_pos_cross_vel[step][dof])
            for i in range(Configuration.NUM_DIMS_3D):
                self.model.addCons(self.f_vel[step][dof][i] == self.crb_mass[step][dof] * self.linear_jacobian[step][dof][i] - self.crb_pos_cross_ang[step][dof][i])

    # Composite Rigid Body algorithm
    def _crb(self, step: int):
        self._link_cinr(step)

        self._reverse_cumulative_sum_matrix(self.crb_pos[step], self.link_cinr_pos[step])
        self._reverse_cumulative_sum_matrix(self.crb_inertial[step], self.link_cinr_inertial[step])

        link_mass = self.config.config_state.link_mass[1:]
        self._reverse_cumulative_sum_scalar(self.crb_mass[step], link_mass)

    # Link Center of INeRtia
    def _link_cinr(self, step: int):
        link_mass = self.config.config_state.link_mass[1:]

        self._link_cinr_pos(step, link_mass)
        self._link_inertial_pos(step)
        self._link_cinr_inertial(step, link_mass)

    def _link_cinr_pos(self, step: int, link_mass):
        self._scale(self.link_cinr_pos[step], link_mass, self.link_inertial_pos[step])

    def _link_inertial_pos(self, step: int):
        link_inertial_pos = self.config.config_state.link_inertial_pos[1:]

        for dof in range(self.config.dofs):
            self._rotate_by_quat(self.link_inertial_pos_rotated[step][dof], link_inertial_pos[dof], self.link_quat[step][dof])

        self.model.addMatrixCons(self.link_inertial_pos[step] == self.link_inertial_pos_rotated[step] + self.link_pos[step] - self.COM[step])

    def _link_cinr_inertial(self, step: int, link_mass):
        self._link_inertial_quat_rotation(step)
        self._link_inertia_hhT_mass_product(step, link_mass)

        link_inertia = self.config.config_state.link_inertia[1:]

        for dof in range(self.config.dofs):
            link_inertia_rotated = self.link_inertial_quat_rotation[step][dof] @ link_inertia[dof] @ self.link_inertial_quat_rotation[step][dof].T
            self.model.addMatrixCons(self.link_cinr_inertial[step][dof] == link_inertia_rotated + self.link_inertia_hhT_mass_product[step][dof])

    def _link_inertial_quat_rotation(self, step: int):
        self._link_inertial_quat(step)
        for dof in range(self.config.dofs):
            self._rotation_from_quat(self.link_inertial_quat_rotation[step][dof], self.link_inertial_quat[step][dof])

    def _link_inertial_quat(self, step):
        link_inertial_quat = self.config.config_state.link_inertial_quat[1:]
        for dof in range(self.config.dofs):
            # link_inertial_quat_rotation = link_quat_rotation @ link_inertial_quat_rotation
            self._compose_quat(self.link_inertial_quat[step][dof], link_inertial_quat[dof], self.link_quat[step][dof])

    def _link_inertia_hhT_mass_product(self, step: int, link_mass):
        self._link_inertia_hhT(step)
        for dof in range(self.config.dofs):
            for i in range(Configuration.NUM_DIMS_3D):
                for j in range(Configuration.NUM_DIMS_3D):
                    self.model.addCons(self.link_inertia_hhT_mass_product[step][dof][i][j] == link_mass[dof] * self.link_inertia_hhT[step][dof][i][j])

    def _link_inertia_hhT(self, step: int):
        for dof in range(self.config.dofs):
            self._hhT(self.link_inertia_hhT[step][dof], self.link_inertial_pos[step][dof])

    def _hhT(self, hhT: MatrixVariable, pos: MatrixVariable):
        for j in range(Configuration.NUM_DIMS_3D):
            for k in range(Configuration.NUM_DIMS_3D):
                if j == k:
                    self.model.addCons(hhT[j][k] == quicksum(pos[l] * pos[l] for l in range(Configuration.NUM_DIMS_3D) if l != j))
                else:
                    self.model.addCons(hhT[j][k] == -pos[j] * pos[k])

    def _applied_force(self, step: int):
        '''
        Compute the force applied by the motors on the joint, via position based PD controller and clamped to stay within min/max force range.
        '''
        self.model.addMatrixCons(self.control_force[step] == self.Kp * (np.array(self.config.config_state.control_pos) - self.pos[step]) - self.Kv * self.vel[step])

        for dof in range(self.config.dofs):
            self.model.addConsIndicator(self.control_force[step][dof] >= self.max_force[dof], binvar=self.control_force_higher_max[step][dof])
            self.model.addConsIndicator(self.control_force[step][dof] <= self.max_force[dof], binvar=self.control_force_higher_max[step][dof], activeone=False)

            self.model.addConsIndicator(self.control_force[step][dof] <= self.min_force[dof], binvar=self.control_force_lower_min[step][dof])
            self.model.addConsIndicator(self.control_force[step][dof] >= self.min_force[dof], binvar=self.control_force_lower_min[step][dof], activeone=False)

            self.model.addConsIndicator(self.control_force_higher_max[step][dof] + self.control_force_lower_min[step][dof] <= 0, binvar=self.control_force_within_range[step][dof])
            self.model.addConsIndicator(self.control_force_higher_max[step][dof] + self.control_force_lower_min[step][dof] >= 1, binvar=self.control_force_within_range[step][dof], activeone=False)

        self.model.addMatrixCons(self.applied_force[step] == self.control_force_within_range[step] * self.control_force[step] + self.min_force * self.control_force_lower_min[step] + self.max_force * self.control_force_higher_max[step])

    def _bias_force(self, step: int):
        '''
        Compute the bias force on the joint, via the bias force formula: bias_force = angular_jacobian . link_torque + linear_jacobian . link_force.
        '''
        self._link_force_torque(step)

        self._scalar_product(self.bias_force_angular[step], self.angular_jacobian[step], self.link_torque[step])
        self._scalar_product(self.bias_force_linear[step], self.linear_jacobian[step], self.link_force[step])
        self.model.addMatrixCons(self.bias_force[step] == self.bias_force_angular[step] + self.bias_force_linear[step])

    def _force(self, step: int):
        self.model.addMatrixCons(self.force[step] == self.applied_force[step] - self.bias_force[step])

    def _solve_mass(self, step: int):
        '''
        Compute the acceleration of the joint, via Newton's second law: F = m * a.
        '''
        self.model.addMatrixCons(self.force[step] == self.mass[step] @ self.acc[step+1])

    def _link_quat_pos(self, step: int):
        self._link_pos0(step)
        self._link_rotation_vector_quat(step)

        link_initial_pos = self.config.config_state.link_initial_pos[1:]
        link_initial_quat = self.config.config_state.link_initial_quat[1:]

        for dof in range(self.config.dofs):
            if dof == 0:
                self.model.addMatrixCons(self.link_quat0[step][dof] == np.array(link_initial_quat[dof]))

                self.model.addMatrixCons(self.link_relative_pos[step][dof] == np.array(link_initial_pos[dof]))
                self.model.addMatrixCons(self.link_pos[step][dof] == self.link_relative_pos[step][dof])
            else:
                self._compose_quat(self.link_quat0[step][dof], link_initial_quat[dof], self.link_quat[step][dof-1])

                self._rotate_by_quat(self.link_relative_pos[step][dof], link_initial_pos[dof], self.link_quat[step][dof-1])
                self.model.addMatrixCons(self.link_pos[step][dof] == self.link_relative_pos[step][dof] + self.link_pos[step][dof-1])

            self._compose_quat(self.link_quat[step][dof], self.link_rotation_vector_quat[step][dof], self.link_quat0[step][dof])

    def _link_pos0(self, step: int):
        self.model.addMatrixCons(self.link_pos0[step] == self.link_pos[step])

    def _link_rotation_vector_quat(self, step: int):
        self._link_rotation_vector(step)
        for dof in range(self.config.dofs):
            self._rotation_vector_quat(self.link_rotation_vector_quat[step][dof], self.link_rotation_vector[step][dof])

    def _link_rotation_vector(self, step: int):
        rotation_axis = self.config.config_state.joint_axis
        self._scale(self.link_rotation_vector[step], self.pos[step], rotation_axis)

    def _linear_angular_jacobian(self, step: int):
        for dof in range(self.config.dofs):
            self._rotate_by_quat(self.xaxis[step][dof], self.config.config_state.joint_axis[dof], self.link_quat0[step][dof])

        self.model.addMatrixCons(self.xanchor[step] == self.link_pos0[step])

        self.model.addMatrixCons(self.offset_pos[step] == self.COM[step] - self.xanchor[step])

        self._cross_product(self.linear_jacobian[step], self.xaxis[step], self.offset_pos[step])
        self.model.addMatrixCons(self.angular_jacobian[step] == self.xaxis[step])

    def _link_force_torque(self, step: int):
        self._f1(step)
        self._f2(step)

        self.model.addMatrixCons(self.link_force_individual[step] == self.f1_vel[step] + self.f2_vel[step])
        self.model.addMatrixCons(self.link_torque_individual[step] == self.f1_ang[step] + self.f2_ang[step])

        for dof in range(self.config.dofs):
            if dof == self.config.dofs - 1:
                self.model.addMatrixCons(self.link_force[step][dof] == self.link_force_individual[step][dof])
                self.model.addMatrixCons(self.link_torque[step][dof] == self.link_torque_individual[step][dof])
            else:
                self.model.addMatrixCons(self.link_force[step][dof] == self.link_force_individual[step][dof] + self.link_force[step][dof+1])
                self.model.addMatrixCons(self.link_torque[step][dof] == self.link_torque_individual[step][dof] + self.link_torque[step][dof+1])

    def _f1(self, step: int):
        self._link_acc(step)

        self._matrix_multiply(self.f1_ang_mat_mul[step], self.link_cinr_inertial[step], self.link_angular_acc[step])

        self._cross_product(self.f1_ang_cross_prod[step], self.link_cinr_pos[step], self.link_linear_acc[step])
        self.model.addMatrixCons(self.f1_ang[step] == self.f1_ang_mat_mul[step] + self.f1_ang_cross_prod[step])

        link_mass = self.config.config_state.link_mass[1:]
        self._scale(self.f1_vel_constant_prod[step], link_mass, self.link_linear_acc[step])

        self._cross_product(self.f1_vel_cross_prod[step], self.link_cinr_pos[step], self.link_angular_acc[step])
        self.model.addMatrixCons(self.f1_vel[step] == self.f1_vel_constant_prod[step] - self.f1_vel_cross_prod[step])

    def _f2(self, step: int):
        self._link_vel(step)

        link_mass = self.config.config_state.link_mass[1:]
        self._scale(self.f2_vel_vel_constant_prod[step], link_mass, self.link_linear_vel[step])
        self._cross_product(self.f2_vel_vel_cross_prod[step], self.link_cinr_pos[step], self.link_angular_vel[step])
        self.model.addMatrixCons(self.f2_vel_vel[step] == self.f2_vel_vel_constant_prod[step] - self.f2_vel_vel_cross_prod[step])

        self._cross_product(self.f2_vel[step], self.link_angular_vel[step], self.f2_vel_vel[step])

        self._matrix_multiply(self.f2_ang_vel_mat_mul[step], self.link_cinr_inertial[step], self.link_angular_vel[step])
        self._cross_product(self.f2_ang_vel_cross_prod[step], self.link_cinr_pos[step], self.link_linear_vel[step])
        self.model.addMatrixCons(self.f2_ang_vel[step] == self.f2_ang_vel_mat_mul[step] + self.f2_ang_vel_cross_prod[step])

        self._cross_product(self.f2_ang_cross_prod1[step], self.link_angular_vel[step], self.f2_ang_vel[step])
        self._cross_product(self.f2_ang_cross_prod2[step], self.link_linear_vel[step], self.f2_vel_vel[step])
        self.model.addMatrixCons(self.f2_ang[step] == self.f2_ang_cross_prod1[step] + self.f2_ang_cross_prod2[step])

    def _link_acc(self, step: int):
        self._joint_linear_angular_jacobian_acc(step)

        self._scale(self.link_linear_acc_individual[step], self.vel[step], self.joint_linear_jacobian_acc[step])
        self._scale(self.link_angular_acc_individual[step], self.vel[step], self.joint_angular_jacobian_acc[step])

        for dof in range(self.config.dofs):
            if dof == 0:
                self.model.addMatrixCons(self.link_linear_acc[step][dof] == self.link_linear_acc_individual[step][dof] + self.config.config_state.gravity)
                self.model.addMatrixCons(self.link_angular_acc[step][dof] == self.link_angular_acc_individual[step][dof])
            else:
                self.model.addMatrixCons(self.link_linear_acc[step][dof] == self.link_linear_acc_individual[step][dof] + self.link_linear_acc[step][dof-1])
                self.model.addMatrixCons(self.link_angular_acc[step][dof] == self.link_angular_acc_individual[step][dof] + self.link_angular_acc[step][dof-1])

    def _link_vel(self, step: int):
        self._scale(self.link_linear_vel_individual[step], self.vel[step], self.linear_jacobian[step])
        self._scale(self.link_angular_vel_individual[step], self.vel[step], self.angular_jacobian[step])

        for dof in range(self.config.dofs):
            if dof == 0:
                self.model.addMatrixCons(self.link_linear_vel[step][dof] == self.link_linear_vel_individual[step][dof])
                self.model.addMatrixCons(self.link_angular_vel[step][dof] == self.link_angular_vel_individual[step][dof])
            else:
                self.model.addMatrixCons(self.link_linear_vel[step][dof] == self.link_linear_vel_individual[step][dof] + self.link_linear_vel[step][dof-1])
                self.model.addMatrixCons(self.link_angular_vel[step][dof] == self.link_angular_vel_individual[step][dof] + self.link_angular_vel[step][dof-1])

    def _joint_linear_angular_jacobian_acc(self, step: int):
        for dof in range(self.config.dofs):
            if dof == 0:
                self.model.addMatrixCons(self.joint_linear_jacobian_acc_cross_ang_vel[step][dof] == np.zeros(Configuration.NUM_DIMS_3D))
                self.model.addMatrixCons(self.joint_linear_jacobian_acc_cross_vel_ang[step][dof] == np.zeros(Configuration.NUM_DIMS_3D))

                self.model.addMatrixCons(self.joint_angular_jacobian_acc[step][dof] == np.zeros(Configuration.NUM_DIMS_3D))
            else:
                self._cross_product_dof(self.joint_linear_jacobian_acc_cross_ang_vel[step][dof], self.link_angular_vel[step][dof-1], self.linear_jacobian[step][dof])
                self._cross_product_dof(self.joint_linear_jacobian_acc_cross_vel_ang[step][dof], self.link_linear_vel[step][dof-1], self.angular_jacobian[step][dof])

                self._cross_product_dof(self.joint_angular_jacobian_acc[step][dof], self.link_angular_vel[step][dof-1], self.angular_jacobian[step][dof])

        self.model.addMatrixCons(self.joint_linear_jacobian_acc[step] == self.joint_linear_jacobian_acc_cross_ang_vel[step] + self.joint_linear_jacobian_acc_cross_vel_ang[step])

    def _integrate(self, step: int, f: Variable, derivative: Variable):
        self.model.addMatrixCons(f[step+1] == f[step] + derivative[step+1] * self.config.step_dt)

    def _scale(self, scaled: MatrixVariable, scale: MatrixVariable, a: MatrixVariable):
        for dof in range(self.config.dofs):
            self._scale_dof(scaled[dof], scale[dof], a[dof])

    def _scale_dof(self, scaled: MatrixVariable, scale: Variable, a: MatrixVariable):
        for i in range(Configuration.NUM_DIMS_3D):
            self.model.addCons(scaled[i] == scale * a[i])

    def _matrix_multiply(self, m: MatrixVariable, a: MatrixVariable, b: MatrixVariable):
        for dof in range(self.config.dofs):
            self._matrix_multiply_dof(m[dof], a[dof], b[dof])

    def _matrix_multiply_dof(self, m: MatrixVariable, a: MatrixVariable, b: MatrixVariable):
        self.model.addMatrixCons(m == a @ b)

    def _scalar_product(self, sp: MatrixVariable, a: MatrixVariable, b: MatrixVariable):
        for dof in range(self.config.dofs):
            self.model.addCons(sp[dof] == self._scalar_product_dof(a[dof], b[dof]))

    def _scalar_product_dof(self, a: MatrixVariable, b: MatrixVariable):
        return (a*b).sum()

    def _cross_product(self, cp: MatrixVariable, a: MatrixVariable, b: MatrixVariable):
        for dof in range(self.config.dofs):
            self._cross_product_dof(cp[dof], a[dof], b[dof])

    def _cross_product_dof(self, cp: MatrixVariable, a: MatrixVariable, b: MatrixVariable):
        self.model.addCons(cp[0] == a[1] * b[2] - a[2] * b[1])
        self.model.addCons(cp[1] == a[2] * b[0] - a[0] * b[2])
        self.model.addCons(cp[2] == a[0] * b[1] - a[1] * b[0])

    def _rotate_by_quat(self, v: MatrixVariable, u: MatrixVariable, quat: MatrixVariable):
        '''
        Rotate u by the rotation represented by quat.
        Set the result in v.

        v = R @ u
        '''
        self.model.addCons(v[0] == u[0] * (quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2] - quat[3]*quat[3]) +
                                   u[1] * 2 * (quat[1] * quat[2] - quat[0] * quat[3]) +
                                   u[2] * 2 * (quat[1] * quat[3] + quat[0] * quat[2]))
        self.model.addCons(v[1] == u[0] * 2 * (quat[0] * quat[3] + quat[1] * quat[2]) +
                                   u[1] * (quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2] - quat[3]*quat[3]) +
                                   u[2] * 2 * (-quat[0] * quat[1] + quat[2] * quat[3]))
        self.model.addCons(v[2] == u[0] * 2 * (-quat[0] * quat[2] + quat[1] * quat[3]) +
                                   u[1] * 2 * (quat[0] * quat[1] + quat[2] * quat[3]) +
                                   u[2] * (quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3]))

    def _compose_quat(self, quat3: MatrixVariable, quat2: MatrixVariable, quat1: MatrixVariable):
        '''
        quat3 is composed of the rotation by quat1 then by quat2
        in terms of rotation matrix, it is equivalent to R3 = R2 @ R1
        '''
        self.model.addCons(quat3[0] == quat1[0] * quat2[0] - quat1[1] * quat2[1] - quat1[2] * quat2[2] - quat1[3] * quat2[3])
        self.model.addCons(quat3[1] == quat1[0] * quat2[1] + quat1[1] * quat2[0] + quat1[2] * quat2[3] - quat1[3] * quat2[2])
        self.model.addCons(quat3[2] == quat1[0] * quat2[2] - quat1[1] * quat2[3] + quat1[2] * quat2[0] + quat1[3] * quat2[1])
        self.model.addCons(quat3[3] == quat1[0] * quat2[3] + quat1[1] * quat2[2] - quat1[2] * quat2[1] + quat1[3] * quat2[0])

    def _rotation_from_quat(self, rotation: MatrixVariable, quat: MatrixVariable):
        """
        Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

        The rotation matrix is:
        [
            [1.0 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1.0 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1.0 - 2*(xx + yy)],
        ]

        where quat = (w, x, y, z)
        """
        # Extract quaternion components: quat = (w, x, y, z)
        w = quat[0]  # scalar part
        x = quat[1]  # x component
        y = quat[2]  # y component
        z = quat[3]  # z component

        # Row 0: [1.0 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)]
        self.model.addCons(rotation[0][0] == 1.0 - 2 * (y * y + z * z))
        self.model.addCons(rotation[0][1] == 2 * (x * y - w * z))
        self.model.addCons(rotation[0][2] == 2 * (x * z + w * y))

        # Row 1: [2*(xy + wz), 1.0 - 2*(xx + zz), 2*(yz - wx)]
        self.model.addCons(rotation[1][0] == 2 * (x * y + w * z))
        self.model.addCons(rotation[1][1] == 1.0 - 2 * (x * x + z * z))
        self.model.addCons(rotation[1][2] == 2 * (y * z - w * x))

        # Row 2: [2*(xz - wy), 2*(yz + wx), 1.0 - 2*(xx + yy)]
        self.model.addCons(rotation[2][0] == 2 * (x * z - w * y))
        self.model.addCons(rotation[2][1] == 2 * (y * z + w * x))
        self.model.addCons(rotation[2][2] == 1.0 - 2 * (x * x + y * y))

    def _rotation_vector_quat(self, quat: MatrixVariable, rotation_vector: MatrixVariable):
        # simplified norm calculation, based on the assumption that rotation_vector is orthogonal
        rotation_vector_norm = quicksum(abs(rotation_vector[i]) for i in range(Configuration.NUM_DIMS_3D))

        self.model.addCons(quat[0] == cos(rotation_vector_norm/2))
        for i in range(Configuration.NUM_DIMS_3D):
            # TODO: handle division by 0
            self.model.addCons(quat[1+i] == sin(rotation_vector_norm/2) * rotation_vector[i] / rotation_vector_norm)

    def _rotation_vector_quat_0(self, quat: MatrixVariable):
        self.model.addMatrixCons(quat == np.array(ScipSolver.QUAT0))

    def _reverse_cumulative_sum_matrix(self, rcs: MatrixVariable, a: MatrixVariable):
        for dof in reversed(range(self.config.dofs)):
            if dof == self.config.dofs-1:
                self.model.addMatrixCons(rcs[dof] == a[dof])
            else:
                self.model.addMatrixCons(rcs[dof] == a[dof] + rcs[dof+1])

    def _reverse_cumulative_sum_scalar(self, rcs: Variable, a: Variable):
        for dof in reversed(range(self.config.dofs)):
            if dof == self.config.dofs-1:
                self.model.addCons(rcs[dof] == a[dof])
            else:
                self.model.addCons(rcs[dof] == a[dof] + rcs[dof+1])