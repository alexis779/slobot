from dataclasses import dataclass
import numpy as np


@dataclass
class Configuration:
    NUM_DIMS_3D = 3
    NUM_DIMS_QUAT = 4

    dofs: int
    step_dt: float
    max_step: int
    gravity: list[float]
    min_force: list[int]
    max_force: list[int]
    Kp: list[float]
    Kv: list[float]
    control_pos: list[float]
    joint_axis: list[list[float]]
    link_initial_quat: list[list[float]]
    link_initial_pos: list[list[float]]
    link_mass: list[float]
    link_inertia: list[list[list[list[float]]]]
    link_inertial_quat: list[list[list[float]]]
    link_inertial_pos: list[list[list[float]]]
    armature: list[float]

    def get_link_initial_pos(self) -> np.ndarray:
        """Return link_initial_pos as a numpy array, excluding the first row."""
        link_initial_pos_array = np.array(self.link_initial_pos)
        return link_initial_pos_array[1:, :]

    def get_link_initial_quat(self) -> np.ndarray:
        """Return link_initial_quat as a numpy array, excluding the first row."""
        link_initial_quat_array = np.array(self.link_initial_quat)
        return link_initial_quat_array[1:, :]

    def get_link_mass(self) -> np.ndarray:
        """Return link_mass as a numpy array, excluding the first element."""
        link_mass_array = np.array(self.link_mass)
        return link_mass_array[1:]

    def get_link_inertia(self) -> np.ndarray:
        """Return link_inertia as a numpy array, excluding the first row (per-link)."""
        link_inertia_array = np.array(self.link_inertia)
        return link_inertia_array[1:, :, :]

    def get_link_inertial_pos(self) -> np.ndarray:
        """Return link_inertial_pos as a numpy array, excluding the first row."""
        link_inertial_pos_array = np.array(self.link_inertial_pos)
        return link_inertial_pos_array[1:, :]

    def get_link_inertial_quat(self) -> np.ndarray:
        """Return link_inertial_quat as a numpy array, excluding the first row."""
        link_inertial_quat_array = np.array(self.link_inertial_quat)
        return link_inertial_quat_array[1:, :]


rigid_body_configuration = Configuration(
    max_step=0,
    dofs=6,
    step_dt=1e-2,
    gravity=[0, 0, 9.81],
    min_force=[-3.5, -3.5, -3.5, -3.5, -3.5, -3.5],
    max_force=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
    Kp=[50, 50, 50, 50, 50, 50],
    Kv=[5.1281, 5.0018, 4.6663, 4.4980, 4.4731, 4.4728],
    armature=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    control_pos=[-1.5708, -1.5708, 1.5708, 1.5708, -1.5708, 1.5708],
    joint_axis=[
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ],
    link_initial_pos=[
        [0.000000, 0.000000, 0.000000],
        [0.000000, -0.045200, 0.016500],
        [0.000000, 0.102500, 0.030600],
        [0.000000, 0.112570, 0.028000],
        [0.000000, 0.005200, 0.134900],
        [0.000000, -0.060100, 0.000000],
        [-0.020200, -0.024400, 0.000000]
    ],
    link_initial_quat=[
        [1.000000, 0.000000, 0.000000, 0.000000],
        [0.707105, 0.707108, 0.000000, 0.000000],
        [0.707109, 0.707105, 0.000000, 0.000000],
        [0.707109, -0.707105, 0.000000, 0.000000],
        [0.707109, -0.707105, 0.000000, 0.000000],
        [0.707109, 0.000000, 0.707105, 0.000000],
        [0.000000, -0.000004, 1.000000, -0.000004]
    ],
    link_mass=[
        0.562466,
        0.119226,
        0.162409,
        0.147968,
        0.066132,
        0.092986,
        0.020244
    ],
    link_inertia=[
        [[0.000615, 0.000000, 0.000000], [0.000000, 0.000481, 0.000000], [0.000000, 0.000000, 0.000365]],
        [[0.000059, 0.000000, 0.000000], [0.000000, 0.000059, 0.000000], [0.000000, 0.000000, 0.000031]],
        [[0.000213, 0.000000, 0.000000], [0.000000, 0.000167, 0.000000], [0.000000, 0.000000, 0.000070]],
        [[0.000139, 0.000000, 0.000000], [0.000000, 0.000108, 0.000000], [0.000000, 0.000000, 0.000048]],
        [[0.000035, 0.000000, 0.000000], [0.000000, 0.000024, 0.000000], [0.000000, 0.000000, 0.000019]],
        [[0.000050, 0.000000, 0.000000], [0.000000, 0.000046, 0.000000], [0.000000, 0.000000, 0.000027]],
        [[0.000011, 0.000000, 0.000000], [0.000000, 0.000009, 0.000000], [0.000000, 0.000000, 0.000003]]
    ],
    link_inertial_pos=[
        [0.000005, -0.015410, 0.028443],
        [-0.000091, 0.059097, 0.031089],
        [-0.000017, 0.070180, 0.003105],
        [-0.003396, 0.001378, 0.076801],
        [-0.008527, -0.035228, -0.000023],
        [0.005524, -0.028017, 0.000484],
        [-0.001617, -0.030347, 0.000450]
    ],
    link_inertial_quat=[
        [0.289504, 0.645114, -0.645380, 0.288963],
        [0.363978, 0.441169, -0.623108, 0.533504],
        [0.501040, 0.498994, -0.493562, 0.506320],
        [0.701995, 0.078800, 0.064563, 0.704859],
        [-0.052281, 0.705235, 0.054952, 0.704905],
        [0.418360, 0.620891, -0.350644, 0.562599],
        [0.696562, 0.716737, -0.023984, -0.022703],
    ],
)
