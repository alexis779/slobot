from slobot.configuration import Configuration
from slobot.robotic_arm import RoboticArm
from slobot.so_arm_100 import SoArm100
import gradio as gr
import torch

class GradioControlSimApp:
    def __init__(self, robotic_arm: RoboticArm):
        self.robotic_arm = robotic_arm

    def launch(self):
        K_p = self.robotic_arm.genesis.entity.get_dofs_kp().tolist()
        K_v = self.robotic_arm.genesis.entity.get_dofs_kv().tolist()

        limits = self.robotic_arm.genesis.entity.get_dofs_limit()
        min_limit, max_limit = limits
        min_limit = min_limit.tolist()
        max_limit = max_limit.tolist()

        force_range = self.robotic_arm.genesis.entity.get_dofs_force_range()
        min_force, max_force = force_range
        min_force = min_force.tolist()
        max_force = max_force.tolist()

        current_pos = self.robotic_arm.genesis.entity.get_qpos()[0].tolist()

        control_force = self.robotic_arm.genesis.entity.get_dofs_control_force()[0].tolist()

        with gr.Blocks() as app:
            for joint_id, joint_name in enumerate(self.robotic_arm.joint_names):
                with gr.Tab(joint_name):
                    joint_id_number = gr.Number(value=joint_id, visible=False)

                    rad_step = (max_limit[joint_id] - min_limit[joint_id]) / 100

                    goal_pos_slider = gr.Slider(minimum=min_limit[joint_id], maximum=max_limit[joint_id], step=rad_step,
                                                value=0, label="Goal Position", interactive=True)
                    current_pos_slider = gr.Slider(minimum=min_limit[joint_id], maximum=max_limit[joint_id], step=rad_step,
                                                  value=current_pos[joint_id], label="Current Position", interactive=False)
                    control_force_slider = gr.Slider(minimum=min_force[joint_id], maximum=max_force[joint_id], step=1,
                                                     value=control_force[joint_id], label="Control Force", interactive=False)
                    gr.Slider(minimum=1, maximum=100, step=1, value=K_p[joint_id], label="K_P", interactive=False)
                    gr.Slider(minimum=1, maximum=100, step=1, value=K_v[joint_id], label="K_D", interactive=False)

                    goal_pos_slider.change(
                        self.set_goal_position,
                        inputs=[joint_id_number, goal_pos_slider],
                        outputs=[current_pos_slider, control_force_slider]
                    )

        app.launch()

    def set_goal_position(self, joint_id, qpos):
        joint_id = int(joint_id)
        idx = [joint_id]

        qpos = float(qpos)
        qpos = [qpos]

        self.robotic_arm.genesis.entity.control_dofs_position(qpos, dofs_idx_local=idx)
        self.robotic_arm.genesis.step()

        current_pos = self.robotic_arm.genesis.entity.get_qpos(qs_idx_local=idx)
        control_force = self.robotic_arm.genesis.entity.get_dofs_control_force(dofs_idx_local=idx)

        return [current_pos[0], control_force[0]]