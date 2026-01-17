import torch

import genesis as gs

gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink, RigidJoint
from genesis.utils import geom as gu

from slobot.configuration import Configuration

from scipy.spatial.transform import Rotation

class Genesis():
    EXTRINSIC_SEQ = 'xyz'

    HOLD_STEPS = 10

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.vis_mode = kwargs.get('vis_mode', 'visual')

        # pass should_start=False to control the start and build process
        should_start = kwargs.get('should_start', True)
        if should_start:
            self.start()

    def start(self):
        kwargs = self.kwargs

        res = kwargs.get('res', Configuration.VGA)
        camera_pos = (-0.125, -1, 0.25)

        lookat = (0, 0, 0)

        lights = [
            { "type": "directional", "dir": (1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0 },
        ]

        self.step_handler = kwargs.get('step_handler', None)

        show_viewer = kwargs.get('show_viewer', True)

        self.fps = kwargs.get('fps', 24)

        dt = 1 / self.fps

        substeps = kwargs.get('substeps', 10)

        requires_grad = kwargs.get('requires_grad', False)

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options = gs.options.SimOptions(
                dt = dt,
                requires_grad = requires_grad,
                substeps = substeps,
            ),
            rigid_options = gs.options.RigidOptions(
                enable_collision = not requires_grad, # TODO, collision dection is not supported with autograd
                #noslip_iterations = 0,
            ),
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
            profiling_options = gs.options.ProfilingOptions(
                show_FPS       = False,
            ),
        )

        plane = gs.morphs.Plane()
        self.scene.add_entity(
            plane,
            visualize_contact=False,
            vis_mode=self.vis_mode,
        )

        arm_morph = self.parse_robot_configuration(**kwargs)

        self.entity: RigidEntity = self.scene.add_entity(
            arm_morph,
            visualize_contact=False,
            vis_mode=self.vis_mode,
        )

        # Kinematic path
        self.base: RigidLink = self.entity.get_link('Base')
        self.shoulder_pan: RigidJoint = self.entity.get_joint('Rotation')
        self.rotation_pitch: RigidLink = self.entity.get_link('Rotation_Pitch')
        self.shoulder_lift: RigidJoint = self.entity.get_joint('Pitch')
        self.upper_arm: RigidLink = self.entity.get_link('Upper_Arm')
        self.elbow_flex: RigidJoint = self.entity.get_joint('Elbow')
        self.lower_arm: RigidLink = self.entity.get_link('Lower_Arm')
        self.wrist_flex: RigidJoint = self.entity.get_joint('Wrist_Pitch')
        self.wrist_pitch_roll: RigidLink = self.entity.get_link('Wrist_Pitch_Roll')
        self.wrist_roll: RigidJoint = self.entity.get_joint('Wrist_Roll')
        self.fixed_jaw: RigidLink = self.entity.get_link('Fixed_Jaw')
        self.gripper: RigidJoint = self.entity.get_joint('Jaw')
        self.moving_jaw: RigidLink = self.entity.get_link('Moving_Jaw')

        self.camera = self.scene.add_camera(
            res    = res,
            pos    = camera_pos,
            lookat = lookat,
            env_idx = 0,
        )

        should_start = kwargs.get('should_start', True)
        if should_start:
            self.build()

    def build(self, n_envs=1):
        self.scene.build(n_envs=n_envs, env_spacing=(0.5, 0.5))

        self.record = self.kwargs.get('record', False)
        if self.record:
            self.camera.start_recording()

        print("Limits=", self.entity.get_dofs_limit())

        qpos = self.entity.get_qpos()
        print("qpos=", qpos)

        Kp = 50
        Kp = torch.full((Configuration.DOFS,), Kp)
        #self.entity.set_dofs_kp(Kp)
        print("Kp=", self.entity.get_dofs_kp())

        Kv = 8
        Kv = torch.full((Configuration.DOFS,), Kv)
        #self.entity.set_dofs_kv(Kv)
        print("Kd=", self.entity.get_dofs_kv())

        max_force = 14
        max_force = torch.full((Configuration.DOFS,), max_force)
        min_force = -max_force
        self.entity.set_dofs_force_range(min_force, max_force)
        print("Force range=", self.entity.get_dofs_force_range())

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

    def parse_robot_configuration(self, **kwargs):
        mjcf_path = kwargs.get('mjcf_path', Configuration.MJCF_CONFIG)
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
        if self.record:
            self.camera.stop_recording(save_to_filename=f"{Configuration.WORK_DIR}/video.mp4")
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

        current_qpos = current_qpos.to(target_qpos.device)
        diff_qpos = current_qpos - target_qpos
        return torch.norm(diff_qpos)

    def link_translate(self, link: RigidLink, t):
        link_pos = link.get_pos()
        link_quat = link.get_quat()

        # Ensure t has shape (n_envs, 3) by stacking it n_envs times
        t = t.unsqueeze(0).expand(self.scene.n_envs, -1)

        t_world = gu.transform_by_quat(t, link_quat)
        return link_pos + t_world

    def quat_to_euler(self, quat):
        quat = quat.cpu()
        return Rotation.from_quat(quat, scalar_first=True).as_euler(seq=self.EXTRINSIC_SEQ)

    def euler_to_quat(self, euler):
        quat = Rotation.from_euler(self.EXTRINSIC_SEQ, euler).as_quat(scalar_first=True)
        return torch.tensor(quat)

    def draw_arrow(self, link, arrow_t, sphere_radius, sphere_color):
        link_pos = link.get_pos()
        sphere_pos = self.link_translate(link, arrow_t)
        arrow_vec = sphere_pos - link_pos
        self.scene.clear_debug_objects()
        self.scene.draw_debug_arrow(link_pos[0], arrow_vec[0], radius = 0.003, color = (1, 0, 0, 0.5))
        self.scene.draw_debug_sphere(sphere_pos[0], sphere_radius, color = sphere_color)
        return sphere_pos[0]