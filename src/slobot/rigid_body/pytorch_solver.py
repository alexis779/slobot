from dataclasses import asdict

import torch

from slobot.rigid_body.configuration import Configuration, rigid_body_configuration
from slobot.rigid_body.state import ConfigurationState, create_entity_state, from_dict


def make_torch_vector_factory(device: torch.device = None, dtype: torch.dtype = None):
    """Create a vector factory that converts lists to torch tensors."""
    def _factory(data: list):
        return torch.tensor(data, device=device, dtype=dtype)
    return _factory

class PytorchSolver:
    # Numerical epsilon threshold for detecting near-zero quaternions/vectors
    EPS = 1e-8
    QUAT0 = [1.0, 0, 0, 0]
    
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.config: Configuration = rigid_body_configuration
        
        # Initialize entity states using factory function
        self.previous_entity = create_entity_state()
        self.current_entity = create_entity_state()

        # Create ConfigurationState with torch tensors using from_dict
        config_dict = asdict(self.config.config_state)
        torch_factory = make_torch_vector_factory(device=self.device)
        self.config_state: ConfigurationState = from_dict(ConfigurationState, config_dict, torch_factory)

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

    # ----------------------------- basic helpers -----------------------------
    def max_abs_error(self, actual, expected):
        return torch.max(torch.abs(actual - expected)).item()

    def cross_product(self, vecs1, vecs2):
        return torch.cross(vecs1, vecs2, dim=1)

    def scalar_product(self, vecs1, vecs2):
        return torch.einsum('ij,ij->i', vecs1, vecs2)

    def outer_product(self, vec1, vec2):
        return torch.einsum('i,j->ij', vec1, vec2)

    def cumulative_sum(self, vecs):
        return torch.cumsum(vecs, dim=0)

    def reverse_cumulative_sum(self, vecs):
        return torch.flip(torch.cumsum(torch.flip(vecs, dims=[0]), dim=0), dims=[0])

    def multiply_matrix_by_vector(self, m, u):
        return torch.einsum('ijk,ik->ij', m, u)

    def multiply_scalar_by_vector(self, c, u):
        return torch.einsum('i,ij->ij', c, u)

    def multiply_scalar_by_matrix(self, c, m):
        return torch.einsum('i,ijk->ijk', c, m)

    def shift_bottom(self, A):
        return torch.vstack([torch.zeros((1, A.shape[1]), device=self.device), A[:-1]])

    def hhT_batch(self, vecs):
        norms = torch.einsum("ni,ni->n", vecs, vecs)
        outers = torch.einsum("ni,nj->nij", vecs, vecs)
        eye = torch.eye(3, device=self.device).unsqueeze(0)
        return norms[:, None, None] * eye - outers

    def matvec(self, m, v):
        return m @ v

    def linalg_solve(self, m, b):
        return torch.linalg.solve(m, b)

    def clip(self, x, min_v, max_v):
        return torch.clamp(x, min_v, max_v)

    def tile_row(self, row):
        return row.repeat(self.config.dofs, 1)

    # ----------------------------- quaternion ops ----------------------------
    def transform_by_quat(self, vecs, quats):
        """Apply quaternion rotation to vectors.
        
        Args:
            vecs: Tensor of shape (N, 3) or (3,) - vectors to rotate
            quats: Tensor of shape (N, 4) or (4,) - quaternions in (w, x, y, z) format
        
        Returns:
            Rotated vectors with same shape as vecs
        """
        
        # Handle single vector/quaternion case
        vecs_was_1d = vecs.dim() == 1
        quats_was_1d = quats.dim() == 1
        
        if vecs_was_1d:
            vecs = vecs.unsqueeze(0)
        if quats_was_1d:
            quats = quats.unsqueeze(0)
        
        # Ensure same batch size for broadcasting
        batch_size = max(vecs.shape[0], quats.shape[0])
        if vecs.shape[0] == 1:
            vecs = vecs.expand(batch_size, -1)
        if quats.shape[0] == 1:
            quats = quats.expand(batch_size, -1)
        
        # Normalize quaternions, handling near-zero quaternions
        quat_norms = torch.norm(quats, dim=-1, keepdim=True)
        zero_mask = quat_norms.squeeze(-1) < self.EPS
        quat_norms_safe = torch.where(zero_mask.unsqueeze(-1), torch.ones_like(quat_norms), quat_norms)
        quats = quats / quat_norms_safe
        
        # For near-zero quaternions, use identity quaternion [1, 0, 0, 0]
        identity_quat = torch.tensor(PytorchSolver.QUAT0, device=self.device)
        quats = torch.where(zero_mask.unsqueeze(-1), identity_quat.unsqueeze(0).expand_as(quats), quats)
        
        # Apply the new transform implementation
        result = self._tc_transform_by_quat(vecs, quats)
        
        # Restore original shape if input was 1D
        if vecs_was_1d and quats_was_1d:
            result = result.squeeze(0)
        
        return result

    # genesis/utils/geom.py: _tc_transform_by_quat
    def _tc_transform_by_quat(self, v, quat, out=None):
        if out is None:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        v_x, v_y, v_z = torch.unbind(v, dim=-1)
        q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)
        q_ww, q_wx, q_wy, q_wz = torch.unbind(q_w * quat, -1)
        q_xx, q_xy, q_xz = torch.unbind(q_x * quat[..., 1:], -1)
        q_yy, q_yz = torch.unbind(q_y * quat[..., 2:], -1)
        q_zz = q_z[..., 0] * quat[..., 3]

        out[..., 0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
        out[..., 1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
        out[..., 2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

        out /= (q_ww + q_xx + q_yy + q_zz)[..., None]

        return out

    def compose_quat_by_quat(self, quat2, quat1):
        vu = self.outer_product(quat1, quat2)
        w = vu[0, 0] - vu[1, 1] - vu[2, 2] - vu[3, 3]
        x = vu[0, 1] + vu[1, 0] + vu[2, 3] - vu[3, 2]
        y = vu[0, 2] - vu[1, 3] + vu[2, 0] + vu[3, 1]
        z = vu[0, 3] + vu[1, 2] - vu[2, 1] + vu[3, 0]
        return torch.stack([w, x, y, z])

    def compose_quat_by_quat_batch(self, quat2, quat1):
        w1, x1, y1, z1 = quat1.T
        w2, x2, y2, z2 = quat2.T
        w = w2*w1 - x2*x1 - y2*y1 - z2*z1
        x = w2*x1 + x2*w1 + y2*z1 - z2*y1
        y = w2*y1 - x2*z1 + y2*w1 + z2*x1
        z = w2*z1 + x2*y1 - y2*x1 + z2*w1
        return torch.stack([w, x, y, z], dim=1)

    def rotation_vector_to_quat(self, rotation_vectors):
        """Convert rotation vectors (axis-angle representation) to quaternions.
        
        Args:
            rotation_vectors: Tensor of shape (N, 3) or (3,) - rotation vectors (axis * angle)
        
        Returns:
            Quaternions in (w, x, y, z) format, shape (N, 4) or (4,)
        """
        
        # Handle single vector case
        if rotation_vectors.dim() == 1:
            rotation_vectors = rotation_vectors.unsqueeze(0)
            single_vec = True
        else:
            single_vec = False
        
        # Compute angle (magnitude of rotation vector)
        angles = torch.norm(rotation_vectors, dim=-1, keepdim=True)
        
        # Handle zero rotation case
        zero_mask = angles.squeeze(-1) < self.EPS
        angles_safe = torch.where(zero_mask.unsqueeze(-1), torch.ones_like(angles), angles)
        
        # Normalize axis
        axes = rotation_vectors / angles_safe
        
        # Compute quaternion: q = [cos(θ/2), sin(θ/2) * axis]
        half_angles = angles_safe / 2.0
        w = torch.cos(half_angles).squeeze(-1)
        sin_half = torch.sin(half_angles)
        xyz = sin_half * axes
        
        # For zero rotations, return identity quaternion [1, 0, 0, 0]
        w = torch.where(zero_mask, torch.ones_like(w), w)
        xyz = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(xyz), xyz)
        
        # Stack into quaternions: (N, 4) in (w, x, y, z) format
        quat = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
        
        if single_vec:
            quat = quat.squeeze(0)
        
        return quat

    def quat_to_rotation_matrix(self, quat):
        """Convert quaternions to rotation matrices.
        
        Args:
            quat: Tensor of shape (N, 4) or (4,) - quaternions in (w, x, y, z) format
        
        Returns:
            Rotation matrices of shape (N, 3, 3) or (3, 3)
        """
        
        # Handle single quaternion case
        if quat.dim() == 1:
            quat = quat.unsqueeze(0)
            single_quat = True
        else:
            single_quat = False
        
        # Normalize quaternions, handling near-zero quaternions
        quat_norms = torch.norm(quat, dim=-1, keepdim=True)
        zero_mask = quat_norms.squeeze(-1) < self.EPS
        quat_norms_safe = torch.where(zero_mask.unsqueeze(-1), torch.ones_like(quat_norms), quat_norms)
        quat = quat / quat_norms_safe
        
        # For near-zero quaternions, use identity quaternion [1, 0, 0, 0]
        identity_quat = torch.tensor(PytorchSolver.QUAT0, device=self.device)
        quat = torch.where(zero_mask.unsqueeze(-1), identity_quat.unsqueeze(0).expand_as(quat), quat)
        
        # Extract quaternion components (w, x, y, z)
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Precompute terms
        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        
        # Build rotation matrices
        R00 = 1 - 2 * (y2 + z2)
        R01 = 2 * (xy - wz)
        R02 = 2 * (xz + wy)
        R10 = 2 * (xy + wz)
        R11 = 1 - 2 * (x2 + z2)
        R12 = 2 * (yz - wx)
        R20 = 2 * (xz - wy)
        R21 = 2 * (yz + wx)
        R22 = 1 - 2 * (x2 + y2)
        
        # Stack into rotation matrices: (N, 3, 3)
        R = torch.stack([
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1)
        ], dim=-2)
        
        if single_quat:
            R = R.squeeze(0)
        
        return R

    # ------------------------- forward-kinematics utils ----------------------
    def compute_link_quat_pos(self, pos):
        dofs = self.config.dofs
        # Initialize output lists to accumulate results
        link_quat_list = []
        link_pos_list = []
        link_quat0_list = []

        # Initialize base values (these are read-only references)
        link_initial_quat = self.config_state.link_initial_quat_no_base
        link_initial_pos = self.config_state.link_initial_pos_no_base
        joint_axis = self.config_state.joint_axis
        axis = self.multiply_scalar_by_vector(pos, joint_axis)
        link_rotation_vector_quat = self.rotation_vector_to_quat(axis)

        for i in range(dofs):
            if i == 0:
                # First link: use initial values directly
                current_link_quat0 = link_initial_quat[i]
                current_link_pos = link_initial_pos[i]
            else:
                # Compute transformed values based on previous link's quaternion
                current_link_quat0 = self.compose_quat_by_quat(link_initial_quat[i], link_quat_list[i-1])
                current_link_rel_pos = self.transform_by_quat(link_initial_pos[i], link_quat_list[i-1])
                current_link_pos = current_link_rel_pos + link_pos_list[i-1]

            # Compute link quaternion from rotation vector quat and base quat
            current_link_quat = self.compose_quat_by_quat(link_rotation_vector_quat[i], current_link_quat0)

            # Store results
            link_quat_list.append(current_link_quat)
            link_pos_list.append(current_link_pos)
            link_quat0_list.append(current_link_quat0)

        # Stack results into tensors
        link_quat = torch.stack(link_quat_list, dim=0)
        link_pos = torch.stack(link_pos_list, dim=0)
        link_quat0 = torch.stack(link_quat0_list, dim=0)

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

        xaxis = self.compute_xaxis(self.config_state.joint_axis, link_quat0)

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
        link_linear_acc = self.config_state.gravity + self.cumulative_sum(link_linear_acc_individual)

        link_angular_acc_individual = self.multiply_scalar_by_vector(vel0, joint_angular_jacobian_acc)
        link_angular_acc = self.cumulative_sum(link_angular_acc_individual)

        f1_ang = self.multiply_matrix_by_vector(link_cinr_inertia, link_angular_acc) + self.cross_product(link_cinr_pos, link_linear_acc)

        f1_vel = self.multiply_scalar_by_vector(self.config_state.link_mass_no_base, link_linear_acc) - self.cross_product(link_cinr_pos, link_angular_acc)

        return f1_vel, f1_ang, link_linear_acc, link_angular_acc, link_linear_acc_individual, link_angular_acc_individual

    def compute_f2(self, link_inertia, link_cinr_pos, linear_jacobian, angular_jacobian, vel0):
        link_linear_vel_individual = self.multiply_scalar_by_vector(vel0, linear_jacobian)
        link_linear_vel = self.cumulative_sum(link_linear_vel_individual)
        link_angular_vel_individual = self.multiply_scalar_by_vector(vel0, angular_jacobian)
        link_angular_vel = self.cumulative_sum(link_angular_vel_individual)
        f2_vel_vel = self.multiply_scalar_by_vector(self.config_state.link_mass_no_base, link_linear_vel) - self.cross_product(link_cinr_pos, link_angular_vel)
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
        link_inertial_quat = self.compose_quat_by_quat_batch(link_quat, self.config_state.link_inertial_quat_no_base)

        rotation = self.quat_to_rotation_matrix(link_inertial_quat)
        rotation_t = rotation.transpose(-2, -1)

        link_cinr_inertial = rotation @ self.config_state.link_inertia_no_base @ rotation_t

        link_inertial_pos = self.transform_by_quat(self.config_state.link_inertial_pos_no_base, link_quat)
        link_inertial_pos = link_inertial_pos + link_pos
        link_inertial_pos = link_inertial_pos - COM

        link_cinr_pos = self.multiply_scalar_by_vector(self.config_state.link_mass_no_base, link_inertial_pos)

        hhT = self.hhT_batch(link_inertial_pos)
        hhT_mass = self.multiply_scalar_by_matrix(self.config_state.link_mass_no_base, hhT)
        link_cinr_inertial = link_cinr_inertial + hhT_mass

        return link_cinr_inertial, link_cinr_pos, link_inertial_pos

    def compute_force(self, bias_force, applied_force):
        return -bias_force + applied_force

    def compute_bias_force(self, link_torque, link_force, angular_jacobian, linear_jacobian):
        bias_force_angular = self.scalar_product(angular_jacobian, link_torque)
        bias_force_linear = self.scalar_product(linear_jacobian, link_force)
        return bias_force_angular + bias_force_linear, bias_force_angular, bias_force_linear

    def compute_applied_force(self, pos0, vel0):
        control_force = self.config_state.Kp * (self.config_state.control_pos - pos0) - self.config_state.Kv * vel0
        applied_force = self.clip(control_force, self.config_state.min_force, self.config_state.max_force)
        return control_force, applied_force

    def compute_newton_euler(self, mass, force, pos0, vel0):
        acc = self.linalg_solve(mass, force)
        step_dt = self.config.step_dt
        vel = vel0 + acc * step_dt
        pos = pos0 + vel * step_dt
        return acc, vel, pos

    def compute_COM(self, link_quat, link_pos):
        # Prepend [1, 0, 0, 0] to link_quat and (0, 0, 0) to link_pos
        link_quat = torch.vstack([torch.tensor(PytorchSolver.QUAT0, device=self.device), link_quat])
        link_pos = torch.vstack([torch.tensor([0, 0, 0], device=self.device), link_pos])

        # Use full versions (including base link) from config_state
        link_inertial_pos_full = self.config_state.link_inertial_pos
        link_mass_full = self.config_state.link_mass
        i_pos = self.transform_by_quat(link_inertial_pos_full, link_quat) + link_pos
        return torch.sum(self.multiply_scalar_by_vector(link_mass_full, i_pos), dim=0) / torch.sum(link_mass_full)

    def compute_f_ang_vel(self, expected_crb_pos, expected_crb_inertial, expected_crb_mass,
                                expected_angular_jacobian, expected_linear_jacobian):
        expected_f_ang = self.multiply_matrix_by_vector(expected_crb_inertial, expected_angular_jacobian) + self.cross_product(expected_crb_pos, expected_linear_jacobian)
        expected_f_vel = self.multiply_scalar_by_vector(expected_crb_mass, expected_linear_jacobian) - self.cross_product(expected_crb_pos, expected_angular_jacobian)
        return expected_f_ang, expected_f_vel

    def compute_mass_matrix(self, expected_f_ang, expected_f_vel, angular_jacobian, linear_jacobian):
        mass_matrix = expected_f_ang @ angular_jacobian.T + expected_f_vel @ linear_jacobian.T

        # update the lower triangular part with the upper triangular part
        mass_matrix = torch.triu(mass_matrix) + torch.triu(mass_matrix, diagonal=1).T

        # add armature
        mass_matrix += torch.diag(self.config_state.armature)

        # discount force jacobian for implicit integration
        # M @ delta_vel = force_{t+1} * delta_t
        #             = (force_t + force_jacobian @ delta_vel) * delta_t
        # (M - delta_t * force_jacobian) @ delta_vel = force_t * delta_t
        force_jacobian = -torch.diag(self.config_state.Kv)
        mass_matrix -= self.config.step_dt * force_jacobian

        return mass_matrix

    def compute_crb(self, expected_cinr_pos, expected_cinr_inertial):
        expected_crb_pos = self.reverse_cumulative_sum(expected_cinr_pos)
        expected_crb_inertial = self.reverse_cumulative_sum(expected_cinr_inertial)

        expected_crb_mass = self.reverse_cumulative_sum(self.config_state.link_mass_no_base)

        return expected_crb_pos, expected_crb_inertial, expected_crb_mass

