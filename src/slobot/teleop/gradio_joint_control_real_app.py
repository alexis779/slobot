from slobot.configuration import Configuration
from slobot.feetech import Feetech
import gradio as gr


class GradioJointControlRealApp:
    def __init__(self):
        self.feetech = Feetech()

    def launch(self):
        self.current_pos = self.feetech.get_pos()
        control_force = self.feetech.get_dofs_control_force()
        torque_enabled = self.feetech.get_torque()
        K_p = self.feetech.get_dofs_kp()
        K_v = self.feetech.get_dofs_kv()
        K_i = self.feetech.get_dofs_ki()

        with gr.Blocks(title="Real Joint Controller") as app:
            gr.Markdown("# Real Joint Controller")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Video(
                        label="Webcam",
                        sources=["webcam"],
                        webcam_options=gr.WebcamOptions(mirror=False),
                        include_audio=False
                    )

                with gr.Column(scale=2):
                    for joint_id, joint_name in enumerate(Configuration.JOINT_NAMES):
                        with gr.Tab(joint_name):
                            joint_id_number = gr.Number(value=joint_id, visible=False)
                            goal_pos_slider = gr.Slider(minimum=0, maximum=self.feetech.model_resolution - 1, step=1,
                                                        value=self.current_pos[joint_id], label="Goal Position", interactive=True)
                            current_pos_slider = gr.Slider(minimum=0, maximum=self.feetech.model_resolution - 1, step=1,
                                                          value=self.current_pos[joint_id], label="Current Position", interactive=False)
                            control_force_slider = gr.Slider(minimum=-1000, maximum=1000, step=1,
                                                             value=control_force[joint_id], label="Control Force", interactive=False)
                            torque_checkbox = gr.Checkbox(value=torque_enabled[joint_id], label="Torque Enabled", interactive=True)
                            gr.Slider(minimum=1, maximum=100, step=1, value=K_p[joint_id], label="K_P", interactive=False)
                            gr.Slider(minimum=1, maximum=100, step=1, value=K_v[joint_id], label="K_D", interactive=False)
                            gr.Slider(minimum=0, maximum=10, step=1, value=K_i[joint_id], label="K_I", interactive=False)

                            goal_pos_slider.change(
                                self.set_goal_position,
                                inputs=[joint_id_number, goal_pos_slider],
                                outputs=[current_pos_slider, control_force_slider, torque_checkbox]
                            )

                            torque_checkbox.change(
                                self.set_torque,
                                inputs=[joint_id_number, torque_checkbox],
                                outputs=[current_pos_slider, control_force_slider, goal_pos_slider]
                            )

        app.launch()

    def set_goal_position(self, joint_id, pos):
        joint_id = int(joint_id)

        self.current_pos[joint_id] = pos

        self.feetech.control_position(self.current_pos)

        control_force = self.feetech.get_dofs_control_force()
        torque_enabled = self.feetech.get_torque()

        return [self.current_pos[joint_id], control_force[joint_id], torque_enabled[joint_id]]

    def set_torque(self, joint_id, torque):
        joint_id = int(joint_id)

        current_torque = self.feetech.get_torque()
        current_torque[joint_id] = torque
        self.feetech.set_torque(current_torque)

        current_pos = self.feetech.get_pos()
        control_force = self.feetech.get_dofs_control_force()
        return [current_pos[joint_id], control_force[joint_id], gr.update(interactive=torque)]
