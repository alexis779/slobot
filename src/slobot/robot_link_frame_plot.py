import numpy as np
import matplotlib.pyplot as plt
from lerobot.model import RobotKinematics
import pinocchio as pin


class RobotLinkFramePlot:
    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str,
    ):
        self.robot_kinematics = RobotKinematics(urdf_path, target_frame_name)

    def plot_link_frames(self, qpos_rad: np.ndarray, axis_length: float = 0.05):
        qpos_deg = np.rad2deg(qpos_rad)
        self.robot_kinematics.forward_kinematics(qpos_deg)
        link_frames = {
            link_name: self.frame(link_name)
            for link_name in self.link_names()
        }

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for child_name, parent_name in self.link_parent_pairs().items():
            if parent_name is None:
                continue
            child_origin = link_frames[child_name][:3, 3]
            parent_origin = link_frames[parent_name][:3, 3]
            ax.plot(
                [parent_origin[0], child_origin[0]],
                [parent_origin[1], child_origin[1]],
                [parent_origin[2], child_origin[2]],
                color="0.4",
                linewidth=1.5,
            )

        label_offset = -0.5 * axis_length
        for link_name, transform in link_frames.items():
            origin = transform[:3, 3]
            rotation = transform[:3, :3]
            ax.scatter(*origin, s=24, color="k")

            for axis_index, color in enumerate("rgb"):
                direction = rotation[:, axis_index] * axis_length
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    color=color,
                    arrow_length_ratio=0.15,
                )

            ax.text(
                origin[0],
                origin[1],
                origin[2] + label_offset,
                link_name,
                fontsize=8,
                ha="center",
                va="top",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        self._set_equal_aspect(ax, link_frames, axis_length)
        fig.tight_layout()
        plt.show()

    def link_names(self) -> list[str]:
        return [
            frame.name
            for frame in self.robot_kinematics.robot.model.frames
            if frame.type == pin.BODY
        ]

    def link_parent_pairs(self) -> dict[str, str | None]:
        model = self.robot_kinematics.robot.model
        body_by_joint = {
            frame.parentJoint: frame.name
            for frame in model.frames
            if frame.type == pin.BODY
        }
        return {
            child_name: body_by_joint.get(model.parents[parent_joint])
            for parent_joint, child_name in body_by_joint.items()
        }

    def frame(self, frame_name: str) -> np.ndarray:
        return self.robot_kinematics.robot.get_T_world_frame(frame_name)

    @staticmethod
    def _set_equal_aspect(ax, link_frames: dict[str, np.ndarray], axis_length: float):
        origins = np.array([transform[:3, 3] for transform in link_frames.values()])
        center = origins.mean(axis=0)
        span = (origins.max(axis=0) - origins.min(axis=0)).max() / 2
        half_range = max(span, axis_length)
        for axis_index, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(center[axis_index] - half_range, center[axis_index] + half_range)
        ax.set_box_aspect([1, 1, 1])
