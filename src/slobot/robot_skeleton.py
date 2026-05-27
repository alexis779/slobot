import numpy as np
from lerobot.model import RobotKinematics
from lerobot.robots.robot import Robot

ORIGIN = np.array([0.0, 0.0, 0.0, 1.0])

URDF_PATH = "../SO-ARM100/Simulation/SO100/so100.urdf"

LINK_NAMES = [
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "jaw",
]

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_joint",
]

# URDF joint limit midpoints (rad), same order as JOINT_NAMES
JOINT_MIDDLE = np.array([
    0.0,
    1.75,
    -1.57079,
    -0.65,
    0.0,
    0.9,
])


class Link:
    def __init__(self, name: str, transform: np.ndarray):
        self.name = name
        self.transform = transform

    @property
    def position(self) -> np.ndarray:
        return (self.transform @ ORIGIN)[:3]


class RobotSkeleton:
    def __init__(
        self,
        robot: Robot,
        urdf_path: str,
        link_names: list[str],
        joint_names: list[str],
    ):
        self.robot = robot
        self.link_names = link_names
        self.joint_names = joint_names
        self.robot_kinematics = RobotKinematics(urdf_path, link_names[0], joint_names)
        self.links: dict[str, Link] = {}

    def _update_kinematics(self) -> dict[str, Link]:
        observation = self.robot.get_observation()
        joint_positions = np.array(
            [observation[f"{key}.pos"] for key in self.robot.bus.motors]
        )
        self.robot_kinematics.forward_kinematics(joint_positions)

        self.links = {
            link_name: Link(link_name, self.robot_kinematics.robot.get_T_world_frame(link_name))
            for link_name in self.link_names
        }

        return self.links

    def get_link_pos(self) -> dict[str, np.ndarray]:
        self._update_kinematics()
        return {name: link.position for name, link in self.links.items()}

    def get_joint_pos(self) -> dict[str, np.ndarray]:
        """Parent link origin for each joint (joint_names[i] is on link_names[i])."""
        self._update_kinematics()
        return {
            joint_name: self.links[self.link_names[joint_index]].position
            for joint_index, joint_name in enumerate(self.joint_names)
        }
