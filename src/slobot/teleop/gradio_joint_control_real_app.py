import math

from slobot.so_arm_100 import SoArm100
from slobot.feetech import Feetech

import gradio as gr


class GradioJointControlRealApp:
    def __init__(self):
        self.robotic_arm = SoArm100()
        self.feetech = Feetech(qpos_handler=self.robotic_arm, qpos_map=self.robotic_arm.qpos_map)

    def round_float(self, value):
        """Round to 1/100 of the value's order of magnitude (scientific-notation exponent)."""
        if value == 0 or not math.isfinite(value):
            return value
        sign = math.copysign(1.0, value)
        v = abs(value)
        exponent = math.floor(math.log10(v))
        step = (10.0 ** exponent) / 100.0
        rounded = round(v / step) * step
        decimals = max(0, 2 - exponent)
        return sign * round(rounded, decimals)

    def get_sim_qpos(self):
        qpos = self.robotic_arm.genesis.entity.get_qpos()[0]
        return [
            self.round_float(qpos[joint_id].item())
            for joint_id in range(len(qpos))
        ]

    def launch(self):
        self.current_pos = self.feetech.get_pos()
        self.feetech.sync_real_to_sim(self.current_pos)
        sim_qpos = self.get_sim_qpos()
        control_force = self.feetech.get_dofs_control_force()
        K_p = self.feetech.get_dofs_kp()
        K_v = self.feetech.get_dofs_kv()

        joint_id_numbers = []
        goal_pos_sliders = []
        current_pos_texts = []
        sim_joint_pos_texts = []
        control_force_texts = []

        max_pos = self.feetech.model_resolution - 1

        with gr.Blocks(title="Real Joint Controller") as app:
            with gr.Row():
                gr.Textbox(value="Joint Control", label=" ", interactive=False, scale=3)
                gr.Textbox(value="Joint Position", label=" ", interactive=False, scale=1)
                gr.Textbox(value="Sim Joint Position", label=" ", interactive=False, scale=1)
                gr.Textbox(value="Control Force", label=" ", interactive=False, scale=1)
                gr.Textbox(value="K_P", label=" ", interactive=False, scale=1)
                gr.Textbox(value="K_D", label=" ", interactive=False, scale=1)

            for joint_id in range(len(sim_qpos)):
                joint_name = self.robotic_arm.joint_names[joint_id]
                joint_id_number = gr.Number(value=joint_id, visible=False)
                joint_id_numbers.append(joint_id_number)

                with gr.Row():
                    goal_pos_slider = gr.Slider(
                        minimum=0,
                        maximum=max_pos,
                        step=1,
                        value=self.current_pos[joint_id],
                        label=joint_name,
                        interactive=True,
                        scale=3,
                    )
                    goal_pos_sliders.append(goal_pos_slider)

                    current_pos_text = gr.Number(
                        value=self.current_pos[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )

                    sim_joint_pos_text = gr.Number(
                        value=sim_qpos[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )

                    control_force_text = gr.Number(
                        value=control_force[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )

                    gr.Number(
                        value=K_p[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )

                    gr.Number(
                        value=K_v[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )

                    current_pos_texts.append(current_pos_text)
                    sim_joint_pos_texts.append(sim_joint_pos_text)
                    control_force_texts.append(control_force_text)

            for joint_id in range(len(sim_qpos)):
                inputs = [joint_id_numbers[joint_id]] + goal_pos_sliders
                goal_pos_sliders[joint_id].change(
                    self.set_goal_position,
                    inputs=inputs,
                    outputs=[
                        current_pos_texts[joint_id],
                        sim_joint_pos_texts[joint_id],
                        control_force_texts[joint_id],
                    ],
                )

        app.launch()

    def set_goal_position(self, joint_id, *qpos):
        joint_id = int(joint_id)

        self.current_pos = [int(qpos_value) for qpos_value in qpos]

        self.feetech.control_position(self.current_pos)

        current_pos = self.feetech.get_pos()
        sim_qpos = self.get_sim_qpos()
        control_force = self.feetech.get_dofs_control_force()

        return [current_pos[joint_id], sim_qpos[joint_id], control_force[joint_id]]

def main() -> None:
    GradioJointControlRealApp().launch()