from dataclasses import dataclass

from slobot.feetech_frame import FeetechFrame

@dataclass
class CameraFrame:
    rgb: any = None
    depth: any = None
    segmentation: any = None
    normal: any = None

@dataclass
class SimulationFrame:
    timestamp: float = None
    control_pos: list[float] = None
    qpos: list[float] = None
    velocity: list[float] = None
    force: list[float] = None
    control_force: list[float] = None
    side_camera_frame: CameraFrame = None
    link_camera_frame: CameraFrame = None
    feetech_frame: FeetechFrame = None

    def frame(self, frame_id):
        match frame_id:
            case 0:
                return self.side_camera_frame.rgb
            case 1:
                return self.side_camera_frame.depth
            case 2:
                return self.side_camera_frame.segmentation
            case 3:
                return self.normal