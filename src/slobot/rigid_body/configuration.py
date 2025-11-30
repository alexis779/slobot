from dataclasses import dataclass

from slobot.rigid_body.state import ConfigurationState


@dataclass
class Configuration:
    NUM_DIMS_3D = 3
    NUM_DIMS_QUAT = 4

    dofs: int
    joint_ids: dict[str, int]
    link_ids: dict[str, int]
    step_dt: float
    max_step: int

    config_state: ConfigurationState

    def get_link_initial_pos(self) -> list[list[float]]:
        """Return link_initial_pos as a list, excluding the first row."""
        return self.config_state.link_initial_pos[1:]

    def get_link_initial_quat(self) -> list[list[float]]:
        """Return link_initial_quat as a list, excluding the first row."""
        return self.config_state.link_initial_quat[1:]

    def get_link_mass(self) -> list[float]:
        """Return link_mass as a list, excluding the first element."""
        return self.config_state.link_mass[1:]

    def get_link_inertia(self) -> list[list[list[list[float]]]]:
        """Return link_inertia as a list, excluding the first row (per-link)."""
        return self.config_state.link_inertia[1:]

    def get_link_inertial_pos(self) -> list[list[list[float]]]:
        """Return link_inertial_pos as a list, excluding the first row."""
        return self.config_state.link_inertial_pos[1:]

    def get_link_inertial_quat(self) -> list[list[list[float]]]:
        """Return link_inertial_quat as a list, excluding the first row."""
        return self.config_state.link_inertial_quat[1:]


rigid_body_configuration = Configuration(
    max_step=0,
    dofs=6,
    joint_ids={
        "shoulder_pan": 0,
        "shoulder_lift": 1,
        "elbow_flex": 2,
        "wrist_flex": 3,
        "wrist_roll": 4,
        "gripper": 5
    },
    link_ids={
        "Rotation_Pitch": 0,
        "Upper_Arm": 1,
        "Lower_Arm": 2,
        "Wrist_Pitch_Roll": 3,
        "Fixed_Jaw": 4,
        "Moving_Jaw": 5
    },
    step_dt=1e-2,
    config_state=ConfigurationState(
        gravity=[0, 0, 9.81],
        middle_pos_offset=[-0.0097,  0.1134,  0.1031,  0.0426,  1.6127,  0.35],
        min_force=[-3.5, -3.5, -3.5, -3.5, -3.5, -3.5],
        max_force=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
        min_dofs_limit=[-1.9200, -3.3200, -0.1740, -1.6600, -2.7900, -0.1740],
        max_dofs_limit=[1.9200, 0.1740, 3.1400, 1.6600, 2.7900, 1.7500],
        Kp=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        Kv=[5.1281, 5.0018, 4.6663, 4.4980, 4.4731, 4.4728],
        armature=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        control_pos=[-1.5708, -1.5708, 1.5708, 1.5708, -1.5708, 1.5708],
        joint_axis=[
            [0, 1.0, 0],
            [1.0, 0, 0],
            [1.0, 0, 0],
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0]
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
    ),
)