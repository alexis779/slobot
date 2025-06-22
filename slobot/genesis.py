import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink, RigidJoint

from slobot.configuration import Configuration

import pprint

from scipy.spatial.transform import Rotation

class Genesis():
    EXTRINSIC_SEQ = 'xyz'

    HOLD_STEPS = 20

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # pass should_start=False to control the start and build process
        should_start = kwargs.get('should_start', True)
        if should_start:
            self.start()

    def start(self):
        kwargs = self.kwargs

        backend = self.backend()
        gs.init(backend=backend)

        vis_mode = 'visual' # 'collision'

        res = kwargs.get('res', Configuration.FHD)
        camera_pos = (-0.125, -1, 0.25)

        lookat = (0, 0, 0)

        lights = [
            { "type": "directional", "dir": (1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0 },
        ]

        self.step_handler = kwargs.get('step_handler', None)

        show_viewer = kwargs.get('show_viewer', True)

        self.fps = kwargs.get('fps', 60)

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options = gs.options.ViewerOptions(
                res           = res,
                camera_lookat = lookat,
                camera_pos    = camera_pos,
                max_FPS       = self.fps,
            ),
            vis_options    = gs.options.VisOptions(
                show_world_frame = False, # True
                lights           = lights,
            ),
            show_FPS       = False,
        )

        plane = gs.morphs.Plane()
        self.scene.add_entity(
            plane,
            vis_mode=vis_mode,
        )

        arm_morph = self.parse_robot_configuration(**kwargs)

        self.entity: RigidEntity = self.scene.add_entity(
            arm_morph,
            vis_mode=vis_mode,
        )

        # TODO errors in non-interactive mode in Docker
        #print("Joints=", pprint.pformat(self.entity.joints))
        #print("Links=", pprint.pformat(self.entity.links))

        self.fixed_jaw: RigidLink = self.entity.get_link('Fixed_Jaw')

        self.jaw: RigidJoint = self.entity.get_joint('Jaw')

        self.camera = self.scene.add_camera(
            res    = res,
            pos    = camera_pos,
            lookat = lookat,
        )

        should_start = kwargs.get('should_start', True)
        if should_start:
            self.build()

    def build(self):
        self.scene.build()

        #self.camera.start_recording()

        print("Limits=", self.entity.get_dofs_limit())

        qpos = self.entity.get_qpos()
        print("qpos=", qpos)

        Kp = 32
        kp_tensor = torch.full_like(qpos, Kp)
        self.entity.set_dofs_kp(kp_tensor)
        print("Kp=", self.entity.get_dofs_kp())

        print("Kd=", self.entity.get_dofs_kv())

        print("Force range=", self.entity.get_dofs_force_range())

        print("collisions=", self.entity.detect_collision())

        damping = self.entity.get_dofs_damping()
        print("damping=", damping)

        stiffness = self.entity.get_dofs_stiffness()
        print("stiffness=", stiffness)

        armature = self.entity.get_dofs_armature()
        print("armature=", armature)

        invweight = self.entity.get_dofs_invweight()
        print("invweight", invweight)

        force = self.entity.get_dofs_force()
        print("force=", force)

        control_force = self.entity.get_dofs_control_force()
        print("control_force=", control_force)

    def backend(self):
        return gs.gpu if torch.cuda.is_available() else gs.cpu

    def parse_robot_configuration(self, **kwargs):
        mjcf_path = kwargs['mjcf_path']
        if mjcf_path is not None:
            return gs.morphs.MJCF(
                file = mjcf_path,
            )

        urdf_path = kwargs['urdf_path']
        if urdf_path is not None:
            return gs.morphs.URDF(
                file  = urdf_path,
                fixed = True,
            )

        raise ValueError(f"Provide either mjcf_path or urdf_path")

    def follow_path(self, target_qpos):
        path = self.entity.plan_path(
            qpos_goal        = target_qpos,
            ignore_collision = True,
            num_waypoints    = self.fps,
        )

        if len(path) == 0:
            return

        for waypoint in path:
            self.entity.control_dofs_position(waypoint)
            self.step()

        # allow more steps to the PD controller to stabilize to the target position
        for _ in range(self.HOLD_STEPS):
            self.step()

        current_error = self.qpos_error(target_qpos)
        print("qpos error=", current_error)

    def stop(self):
        gs.destroy()

    def move(self, link, target_pos, target_quat):
        target_qpos = self.entity.inverse_kinematics(
            link     = link,
            pos      = target_pos,
            quat     = target_quat,
        )

        self.follow_path(target_qpos)
        self.validate_target(self.fixed_jaw, target_pos, target_quat)

    def step(self):
        self.scene.step()
        if self.step_handler is not None:
            self.step_handler.handle_step()

    def hold_entity(self):
        while True:
            self.step()

    def get_qpos_idx(self, joint):
        return joint.idx_local - 1  # offset the base joint, which is not part of qpos variables

    def update_qpos(self, joint, qpos):
        target_qpos = self.entity.get_qpos()
        joint_idx = self.get_qpos_idx(joint)
        target_qpos[joint_idx] = qpos
        self.follow_path(target_qpos)

    def validate_target(self, link, target_pos, target_quat):
        self.validate_pos(link, target_pos)
        self.validate_quat(link, target_quat)

    def validate_pos(self, link, target_pos):
        if target_pos is None:
            return

        current_pos = link.get_pos()

        current_pos = current_pos.to(target_pos.device)
        error = torch.norm(current_pos - target_pos)
        print("pos error=", error)

    def validate_quat(self, link, target_quat):
        if target_quat is None:
            return

        current_quat = link.get_quat()

        current_quat = current_quat.to(target_quat.device)
        error = torch.norm(current_quat - target_quat)
        print("quat error=", error)

    def qpos_error(self, target_qpos):
        current_qpos = self.entity.get_qpos()

        # To avoid division by 0, create a target_qpos_denominator variable where 0 are replaced with 1
        target_qpos_denominator = torch.where(target_qpos == 0, torch.tensor(1.0, device=target_qpos.device), target_qpos)

        current_qpos = current_qpos.to(target_qpos.device)
        error = torch.abs((current_qpos - target_qpos) / target_qpos_denominator)
        return torch.norm(error)

    def translate(self, pos, t, euler):
        r = Rotation.from_euler(self.EXTRINSIC_SEQ, euler)
        t = r.apply(t)
        return pos + t

    def quat_to_euler(self, quat):
        quat = quat.cpu()
        return Rotation.from_quat(quat, scalar_first=True).as_euler(seq=self.EXTRINSIC_SEQ)

    def euler_to_quat(self, euler):
        quat = Rotation.from_euler(self.EXTRINSIC_SEQ, euler).as_quat(scalar_first=True)
        return torch.tensor(quat)

    def draw_arrow(self, link, t, color):
        #self.scene.clear_debug_objects()
        link_pos = link.get_pos()
        t_pos = self.link_translate(link, t)
        arrow_vec = t_pos - link_pos
        self.scene.draw_debug_arrow(link_pos, arrow_vec, radius = 0.003, color=color)

    def link_translate(self, link, t):
        link_pos = link.get_pos()
        link_quat = link.get_quat()
        link_euler = self.quat_to_euler(link_quat)
        return self.translate(link_pos, t, link_euler)