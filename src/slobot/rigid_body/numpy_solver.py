from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation as R

from slobot.rigid_body.configuration import Configuration, rigid_body_configuration
from slobot.rigid_body.state import ConfigurationState, create_entity_state, from_dict


def numpy_vector_factory(data: list):
    """Convert list to numpy array for configuration loading."""
    return np.array(data)

class NumpySolver:
    def __init__(self) -> None:
        self.config: Configuration = rigid_body_configuration
        # Initialize entity states using factory function
        self.previous_entity = create_entity_state()
        self.current_entity = create_entity_state()

        # Create ConfigurationState with numpy arrays using from_dict
        config_dict = asdict(self.config.config_state)
        self.config_state: ConfigurationState = from_dict(ConfigurationState, config_dict, numpy_vector_factory)

        # Drop base link from config_state fields (excluding first element/row)
        self.drop_base_link()

    def drop_base_link(self):
        """Create versions without base link in config_state (excluding first element/row)."""
        # Create versions without base link (excluding first element/row)
        self.config_state.link_initial_quat_no_base = self.config_state.link_initial_quat[1:]
        self.config_state.link_initial_pos_no_base = self.config_state.link_initial_pos[1:]
        self.config_state.link_mass_no_base = self.config_state.link_mass[1:]
        self.config_state.link_inertia_no_base = self.config_state.link_inertia[1:]
        self.config_state.link_inertial_quat_no_base = self.config_state.link_inertial_quat[1:]
        self.config_state.link_inertial_pos_no_base = self.config_state.link_inertial_pos[1:]

    def _list_to_array(self, vec):
        """Convert a list or vector from config to numpy array."""
        return np.array(vec)

    # ----------------------------- basic helpers -----------------------------
    def max_abs_error(self, actual, expected):
        return np.max(np.abs(actual - expected))

    def cross_product(self, vecs1, vecs2):
        return np.cross(vecs1, vecs2, axis=1)

    def scalar_product(self, vecs1, vecs2):
        return np.einsum('ij,ij->i', vecs1, vecs2)

    def outer_product(self, vec1, vec2):
        return np.einsum('i,j->ij', vec1, vec2)

    def cumulative_sum(self, vecs):
        return np.cumsum(vecs, axis=0)

    def reverse_cumulative_sum(self, vecs):
        return np.flip(np.cumsum(np.flip(vecs, axis=0), axis=0), axis=0)

    def multiply_matrix_by_vector(self, m, u):
        return np.einsum('ijk,ik->ij', m, u)

    def multiply_scalar_by_vector(self, c, u):
        return np.einsum('i,ij->ij', c, u)

    def multiply_scalar_by_matrix(self, c, m):
        return np.einsum('i,ijk->ijk', c, m)

    def shift_bottom(self, A):
        return np.vstack([np.zeros((1, A.shape[1])), A[:-1]])

    def hhT_batch(self, vecs):
        norms = np.einsum("ni,ni->n", vecs, vecs)
        outers = np.einsum("ni,nj->nij", vecs, vecs)
        return norms[:, None, None] * np.eye(3) - outers

    def matvec(self, m, v):
        return m @ v

    def linalg_solve(self, m, b):
        return np.linalg.solve(m, b)

    def clip(self, x, min_v, max_v):
        return np.clip(x, min_v, max_v)

    def tile_row(self, row):
        return np.tile(row, (self.config.dofs, 1))

    # ----------------------------- quaternion ops ----------------------------
    def transform_by_quat(self, vecs, quats):
        r = self.quat_to_rotation_matrix(quats)
        return r.apply(vecs)

    def compose_quat_by_quat(self, quat2, quat1):
        vu = self.outer_product(quat1, quat2)
        w = vu[0, 0] - vu[1, 1] - vu[2, 2] - vu[3, 3]
        x = vu[0, 1] + vu[1, 0] + vu[2, 3] - vu[3, 2]
        y = vu[0, 2] - vu[1, 3] + vu[2, 0] + vu[3, 1]
        z = vu[0, 3] + vu[1, 2] - vu[2, 1] + vu[3, 0]
        return np.array([w, x, y, z])

    def compose_quat_by_quat_batch(self, quat2, quat1):
        w1, x1, y1, z1 = quat1.T
        w2, x2, y2, z2 = quat2.T
        w = w2*w1 - x2*x1 - y2*y1 - z2*z1
        x = w2*x1 + x2*w1 + y2*z1 - z2*y1
        y = w2*y1 - x2*z1 + y2*w1 + z2*x1
        z = w2*z1 + x2*y1 - y2*x1 + z2*w1
        return np.stack([w, x, y, z], axis=1)

    def rotation_vector_to_quat(self, rotation_vectors):
        r = R.from_rotvec(rotation_vectors)
        # Return with scalar-first layout (w,x,y,z)
        return r.as_quat(scalar_first=True)

    def quat_to_rotation_matrix(self, quat):
        return R.from_quat(quat, scalar_first=True)

    # ------------------------- forward-kinematics utils ----------------------
    def compute_link_quat_pos(self, pos):
        dofs = self.config.dofs
        link_quat = np.zeros((dofs, Configuration.NUM_DIMS_QUAT))
        link_pos = np.zeros((dofs, Configuration.NUM_DIMS_3D))

        # Copy arrays to avoid mutating config_state
        link_quat0 = self.config_state.link_initial_quat_no_base.copy()
        link_rel_pos = self.config_state.link_initial_pos_no_base.copy()

        joint_axis = self.config_state.joint_axis
        axis = self.multiply_scalar_by_vector(pos, joint_axis)
        link_rotation_vector_quat = self.rotation_vector_to_quat(axis)

        for i in range(dofs):
            if i == 0:
                link_pos[i] = link_rel_pos[i]
            else:
                link_quat0[i] = self.compose_quat_by_quat(link_quat0[i], link_quat[i-1])
                link_rel_pos[i] = self.transform_by_quat(link_rel_pos[i], link_quat[i-1])
                link_pos[i] = link_rel_pos[i] + link_pos[i-1]
            link_quat[i] = self.compose_quat_by_quat(link_rotation_vector_quat[i], link_quat0[i])

        return link_quat, link_pos, link_quat0, link_pos, link_rotation_vector_quat

    def compute_xaxis(self, joint_axis, link_quat0):
        return self.transform_by_quat(joint_axis, link_quat0)

    def compute_linear_and_angular_jacobian(self, xaxis, COM, xanchor):
        angular_jacobian = xaxis
        COM_matrix = self.tile_row(COM)
        linear_jacobian = self.cross_product(xaxis, COM_matrix - xanchor)
        return angular_jacobian, linear_jacobian

    def forward_kinematics(self, pos0):
        link_quat, link_pos, link_quat0, link_pos0, link_rotation_vector_quat = self.compute_link_quat_pos(pos0)

        xanchor = link_pos0

        joint_axis = self.config_state.joint_axis
        xaxis = self.compute_xaxis(joint_axis, link_quat0)

        angular_jacobian = xaxis

        COM = self.compute_COM(link_quat, link_pos)

        angular_jacobian, linear_jacobian = self.compute_linear_and_angular_jacobian(xaxis, COM, xanchor)

        return angular_jacobian, linear_jacobian, link_quat, link_pos, COM

    def forward_dynamics(self, pos0, vel0, linear_jacobian, angular_jacobian, link_quat, link_pos, COM):
        link_cinr_inertial, link_cinr_pos, link_inertial_pos = self.compute_link_inertia(link_quat, link_pos, COM)

        f2_ang, f2_vel, link_angular_vel, link_linear_vel, link_angular_vel_individual, link_linear_vel_individual, f2_vel_vel, f2_ang_vel = self.compute_f2(link_cinr_inertial, link_cinr_pos, linear_jacobian, angular_jacobian, vel0)

        joint_linear_jacobian_acc, joint_angular_jacobian_acc = self.compute_joint_jacobian_acc(link_angular_vel, link_linear_vel, linear_jacobian, angular_jacobian)

        f1_vel, f1_ang, link_linear_acc, link_angular_acc, link_linear_acc_individual, link_angular_acc_individual = self.compute_f1(link_cinr_inertial, link_cinr_pos, joint_linear_jacobian_acc, joint_angular_jacobian_acc, vel0)

        link_force, link_torque = self.compute_link_force_torque(f1_vel, f1_ang, f2_vel, f2_ang)

        bias_force, bias_force_angular, bias_force_linear = self.compute_bias_force(link_torque, link_force, angular_jacobian, linear_jacobian)

        control_force, applied_force = self.compute_applied_force(pos0, vel0)

        force = self.compute_force(bias_force, applied_force)

        return force, link_cinr_pos, link_cinr_inertial

    def mass(self, link_cinr_pos, link_cinr_inertial, angular_jacobian, linear_jacobian):
        crb_pos, crb_inertial, crb_mass = self.compute_crb(link_cinr_pos, link_cinr_inertial)

        f_ang, f_vel = self.compute_f_ang_vel(crb_pos, crb_inertial, crb_mass, angular_jacobian, linear_jacobian)

        mass_matrix = self.compute_mass_matrix(f_ang, f_vel, angular_jacobian, linear_jacobian)

        return mass_matrix

    def step(self):
        # Swap previous and current entity so the next call uses the newly computed values
        self.previous_entity, self.current_entity = self.current_entity, self.previous_entity

        pos0 = self.previous_entity.joint.pos
        vel0 = self.previous_entity.joint.vel

        angular_jacobian, linear_jacobian, link_quat, link_pos, COM = self.forward_kinematics(pos0)

        # Store link state in current_entity
        self.current_entity.link.quat = link_quat
        self.current_entity.link.pos = link_pos

        force, link_cinr_pos, link_cinr_inertial = self.forward_dynamics(pos0, vel0,
                                                                        linear_jacobian, angular_jacobian,
                                                                        link_quat, link_pos,
                                                                        COM)

        mass_matrix = self.mass(link_cinr_pos, link_cinr_inertial, angular_jacobian, linear_jacobian)

        acc, vel, pos = self.compute_newton_euler(mass_matrix, force, pos0, vel0)

        # Store results in current_entity
        self.current_entity.joint.pos = pos
        self.current_entity.joint.vel = vel

    def get_pos(self):
        """Get current position."""
        return self.current_entity.joint.pos

    def get_vel(self):
        """Get current velocity."""
        return self.current_entity.joint.vel

    def set_pos(self, pos):
        """Set current position."""
        self.current_entity.joint.pos = pos

    def set_vel(self, vel):
        """Set current velocity."""
        self.current_entity.joint.vel = vel

    def get_link_quat(self, link_name = None):
        """Get current link quaternion."""

        if link_name is not None:
            link_id = self.config.link_ids[link_name]
            return self.current_entity.link.quat[link_id]

        return self.current_entity.link.quat

    def get_link_pos(self, link_name = None):
        """Get current link position."""

        if link_name is not None:
            link_id = self.config.link_ids[link_name]
            return self.current_entity.link.pos[link_id]

        return self.current_entity.link.pos

    def control_dofs_position(self, pos):
        """Control the position of the DOFs."""
        self.config_state.control_pos = pos

    def compute_joint_jacobian_acc(self, link_angular_vel, link_linear_vel, linear_jacobian, angular_jacobian):
        link_angular_vel_shifted = self.shift_bottom(link_angular_vel)
        link_linear_vel_shifted = self.shift_bottom(link_linear_vel)
        joint_linear_jacobian_acc = self.cross_product(link_angular_vel_shifted, linear_jacobian) + self.cross_product(link_linear_vel_shifted, angular_jacobian)
        joint_angular_jacobian_acc = self.cross_product(link_angular_vel_shifted, angular_jacobian)
        return joint_linear_jacobian_acc, joint_angular_jacobian_acc

    def compute_f1(self, link_cinr_inertia, link_cinr_pos, joint_linear_jacobian_acc, joint_angular_jacobian_acc, vel0):
        link_linear_acc_individual = self.multiply_scalar_by_vector(vel0, joint_linear_jacobian_acc)
        gravity = self.config_state.gravity
        link_linear_acc = gravity + self.cumulative_sum(link_linear_acc_individual)

        link_angular_acc_individual = self.multiply_scalar_by_vector(vel0, joint_angular_jacobian_acc)
        link_angular_acc = self.cumulative_sum(link_angular_acc_individual)

        f1_ang = self.multiply_matrix_by_vector(link_cinr_inertia, link_angular_acc) + self.cross_product(link_cinr_pos, link_linear_acc)

        link_mass = self.config_state.link_mass_no_base
        f1_vel = self.multiply_scalar_by_vector(link_mass, link_linear_acc) - self.cross_product(link_cinr_pos, link_angular_acc)

        return f1_vel, f1_ang, link_linear_acc, link_angular_acc, link_linear_acc_individual, link_angular_acc_individual

    def compute_f2(self, link_inertia, link_cinr_pos, linear_jacobian, angular_jacobian, vel0):
        link_linear_vel_individual = self.multiply_scalar_by_vector(vel0, linear_jacobian)
        link_linear_vel = self.cumulative_sum(link_linear_vel_individual)
        link_angular_vel_individual = self.multiply_scalar_by_vector(vel0, angular_jacobian)
        link_angular_vel = self.cumulative_sum(link_angular_vel_individual)
        link_mass = self.config_state.link_mass_no_base
        f2_vel_vel = self.multiply_scalar_by_vector(link_mass, link_linear_vel) - self.cross_product(link_cinr_pos, link_angular_vel)
        f2_vel = self.cross_product(link_angular_vel, f2_vel_vel)
        f2_ang_vel = self.multiply_matrix_by_vector(link_inertia, link_angular_vel) + self.cross_product(link_cinr_pos, link_linear_vel)
        f2_ang = self.cross_product(link_angular_vel, f2_ang_vel) + self.cross_product(link_linear_vel, f2_vel_vel)
        return f2_ang, f2_vel, link_angular_vel, link_linear_vel, link_angular_vel_individual, link_linear_vel_individual, f2_vel_vel, f2_ang_vel

    def compute_link_force_torque(self, f1_vel, f1_ang, f2_vel, f2_ang):
        link_force_individual = f1_vel + f2_vel
        link_torque_individual = f1_ang + f2_ang

        # traverse the kinematic chain in reverse to get the cumulative forces/torques bottom-up
        link_force = self.reverse_cumulative_sum(link_force_individual)
        link_torque = self.reverse_cumulative_sum(link_torque_individual)
        return link_force, link_torque

    def compute_link_inertia(self, link_quat, link_pos, COM):
        link_inertial_quat = self.config_state.link_inertial_quat_no_base
        link_inertial_quat = self.compose_quat_by_quat_batch(link_quat, link_inertial_quat)

        rotation = self.quat_to_rotation_matrix(link_inertial_quat).as_matrix()

        rotation_t = rotation.transpose(0, 2, 1)

        link_cinr_inertial = self.config_state.link_inertia_no_base
        link_cinr_inertial = rotation @ link_cinr_inertial @ rotation_t

        link_inertial_pos = self.config_state.link_inertial_pos_no_base
        link_inertial_pos = self.transform_by_quat(link_inertial_pos, link_quat)
        link_inertial_pos = link_inertial_pos + link_pos
        link_inertial_pos = link_inertial_pos - COM

        link_mass = self.config_state.link_mass_no_base
        link_cinr_pos = self.multiply_scalar_by_vector(link_mass, link_inertial_pos)

        hhT = self.hhT_batch(link_inertial_pos)
        hhT_mass = self.multiply_scalar_by_matrix(link_mass, hhT)
        link_cinr_inertial = link_cinr_inertial + hhT_mass

        return link_cinr_inertial, link_cinr_pos, link_inertial_pos

    def compute_force(self, bias_force, applied_force):
        return -bias_force + applied_force

    def compute_bias_force(self, link_torque, link_force, angular_jacobian, linear_jacobian):
        bias_force_angular = self.scalar_product(angular_jacobian, link_torque)
        bias_force_linear = self.scalar_product(linear_jacobian, link_force)
        return bias_force_angular + bias_force_linear, bias_force_angular, bias_force_linear

    def compute_applied_force(self, pos0, vel0):
        Kp = self.config_state.Kp
        Kv = self.config_state.Kv
        control_pos = self.config_state.control_pos
        min_force = self.config_state.min_force
        max_force = self.config_state.max_force
        control_force = Kp * (control_pos - pos0) - Kv * vel0
        applied_force = self.clip(control_force, min_force, max_force)
        return control_force, applied_force

    def compute_newton_euler(self, mass, force, pos0, vel0):
        acc = self.linalg_solve(mass, force)
        vel = vel0 + acc * self.config.step_dt
        pos = pos0 + vel * self.config.step_dt
        return acc, vel, pos

    def compute_COM(self, link_quat, link_pos):
        # Prepend [1, 0, 0, 0] to link_quat and (0, 0, 0) to link_pos
        link_quat = np.vstack([np.array([[1, 0, 0, 0]]), link_quat])
        link_pos = np.vstack([np.array([[0, 0, 0]]), link_pos])

        link_inertial_pos = self.config_state.link_inertial_pos
        i_pos = self.transform_by_quat(link_inertial_pos, link_quat) + link_pos
        link_mass = self.config_state.link_mass
        return np.sum(self.multiply_scalar_by_vector(link_mass, i_pos), axis=0) / np.sum(link_mass)

    def compute_f_ang_vel(self, expected_crb_pos, expected_crb_inertial, expected_crb_mass,
                                expected_angular_jacobian, expected_linear_jacobian):
        expected_f_ang = self.multiply_matrix_by_vector(expected_crb_inertial, expected_angular_jacobian) + np.cross(expected_crb_pos, expected_linear_jacobian)
        expected_f_vel = self.multiply_scalar_by_vector(expected_crb_mass, expected_linear_jacobian) - np.cross(expected_crb_pos, expected_angular_jacobian)
        return expected_f_ang, expected_f_vel

    def compute_mass_matrix(self, expected_f_ang, expected_f_vel, angular_jacobian, linear_jacobian):
        mass_matrix = expected_f_ang @ angular_jacobian.T + expected_f_vel @ linear_jacobian.T

        # update the lower triangular part with the upper triangular part
        mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T

        # add armature
        armature = self.config_state.armature
        mass_matrix += np.diag(armature)

        # discount force jacobian for implicit integration
        # M @ delta_vel = force_{t+1} * delta_t
        #             = (force_t + force_jacobian @ delta_vel) * delta_t
        # (M - delta_t * force_jacobian) @ delta_vel = force_t * delta_t
        Kv = self.config_state.Kv
        force_jacobian = -np.diag(Kv)
        mass_matrix -= self.config.step_dt * force_jacobian

        return mass_matrix

    def compute_crb(self, expected_cinr_pos, expected_cinr_inertial):
        expected_crb_pos = self.reverse_cumulative_sum(expected_cinr_pos)
        expected_crb_inertial = self.reverse_cumulative_sum(expected_cinr_inertial)

        link_mass = self.config_state.link_mass_no_base
        expected_crb_mass = self.reverse_cumulative_sum(link_mass)

        return expected_crb_pos, expected_crb_inertial, expected_crb_mass