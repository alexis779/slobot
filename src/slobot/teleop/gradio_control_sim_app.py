from slobot.configuration import Configuration
from slobot.robotic_arm import RoboticArm
from slobot.so_arm_100 import SoArm100
import gradio as gr
import math
import torch

class GradioControlSimApp:
    def __init__(self, robotic_arm: RoboticArm):
        self.robotic_arm = robotic_arm

    def round_float(self, value, decimals=3):
        """Round a float value to specified number of decimals."""
        return round(value, decimals)

    def launch(self):
        K_p = [self.round_float(v) for v in self.robotic_arm.genesis.entity.get_dofs_kp().tolist()]
        K_v = [self.round_float(v) for v in self.robotic_arm.genesis.entity.get_dofs_kv().tolist()]

        limits = self.robotic_arm.genesis.entity.get_dofs_limit()
        min_limit, max_limit = limits

        rad_steps = (max_limit - min_limit) / 100
        rad_steps = [self.round_float(v) for v in rad_steps.tolist()]

        min_limit = [self.round_float(v) for v in min_limit.tolist()]
        max_limit = [self.round_float(v) for v in max_limit.tolist()]

        force_range = self.robotic_arm.genesis.entity.get_dofs_force_range()
        min_force, max_force = force_range
        min_force = [self.round_float(v) for v in min_force.tolist()]
        max_force = [self.round_float(v) for v in max_force.tolist()]

        joint_id_numbers = []
        goal_pos_sliders = []
        current_pos_texts = []
        control_force_texts = []

        with gr.Blocks() as app:
            # Header row
            with gr.Row():
                gr.Textbox(value="Joint Control", label=" ", interactive=False, scale=3)
                gr.Textbox(value="Joint Position", label=" ", interactive=False, scale=1)
                gr.Textbox(value="Control Force", label=" ", interactive=False, scale=1)
                gr.Textbox(value="K_P", label=" ", interactive=False, scale=1)
                gr.Textbox(value="K_D", label=" ", interactive=False, scale=1)

            # One row per joint
            for joint_id, joint_name in enumerate(self.robotic_arm.joint_names):
                # Hidden joint ID number for callbacks
                joint_id_number = gr.Number(value=joint_id, visible=False)
                joint_id_numbers.append(joint_id_number)

                with gr.Row():
                    # Column 1: Control slider (3 columns wide)
                    goal_pos_slider = gr.Slider(
                        minimum=min_limit[joint_id],
                        maximum=max_limit[joint_id],
                        step=rad_steps[joint_id],
                        value=0,
                        label=joint_name,
                        interactive=True,
                        scale=3
                    )
                    goal_pos_sliders.append(goal_pos_slider)

                    # Column 2: Joint position
                    current_pos_text = gr.Number(
                        value=0,
                        label=" ",
                        interactive=False,
                        scale=1
                    )

                    # Column 3: Control force
                    control_force_text = gr.Number(
                        value=0,
                        label=" ",
                        interactive=False,
                        scale=1
                    )

                    # Column 4: K_P
                    gr.Number(
                        value=K_p[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1
                    )

                    # Column 5: K_D
                    gr.Number(
                        value=K_v[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1
                    )

                    current_pos_texts.append(current_pos_text)
                    control_force_texts.append(control_force_text)

            # Setup event handlers
            for joint_id, joint_name in enumerate(self.robotic_arm.joint_names):
                inputs = [joint_id_numbers[joint_id]] + goal_pos_sliders
                goal_pos_sliders[joint_id].change(
                    self.set_goal_position,
                    inputs=inputs,
                    outputs=[current_pos_texts[joint_id], control_force_texts[joint_id]]
                )

        app.launch()

    def set_goal_position(self, joint_id, *qpos):
        joint_id = int(joint_id)

        qpos = [float(qpos_value) for qpos_value in qpos]

        self.robotic_arm.genesis.entity.control_dofs_position(qpos)
        self.robotic_arm.genesis.step()

        current_pos = self.robotic_arm.genesis.entity.get_qpos()
        current_pos = current_pos[0][joint_id].item()
        current_pos = self.round_float(current_pos)
        
        control_force = self.robotic_arm.genesis.entity.get_dofs_control_force()
        control_force = control_force[0][joint_id].item()
        control_force = self.round_float(control_force)

        return [current_pos, control_force]