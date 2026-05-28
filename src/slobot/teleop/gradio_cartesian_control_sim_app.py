import math

from slobot.robotic_arm import RoboticArm
from slobot.teleop.cartesian_sim_controller import CartesianSimController
import gradio as gr

AXIS_ROW_CSS = """
.axis-x { background-color: rgba(255, 0, 0, 0.25) !important; }
.axis-y { background-color: rgba(0, 255, 0, 0.25) !important; }
.axis-z { background-color: rgba(0, 0, 255, 0.25) !important; }
"""


class GradioCartesianControlSimApp:
    def __init__(self, robotic_arm: RoboticArm):
        self.robotic_arm = robotic_arm
        self.controller = CartesianSimController(robotic_arm)
        self.active_motion: tuple[str, int, int] | None = None

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

    def _joint_positions_display(self):
        qpos = self.robotic_arm.genesis.entity.get_qpos()[0].tolist()
        return [self.round_float(v) for v in qpos]

    def _apply_active_motion(self):
        kind, axis_id, direction = self.active_motion
        if kind == "translate":
            self.controller.translate_local(axis_id, direction)
        else:
            self.controller.rotate_local(axis_id, direction)

    def toggle_motion(self, kind: str, axis_id: int, direction: int):
        motion = (kind, axis_id, direction)
        if self.active_motion == motion:
            self.active_motion = None
        else:
            self.active_motion = motion
            self.controller.sync_targets_from_link()
            self._apply_active_motion()
        return self._joint_positions_display()

    def step_motion(self):
        if self.active_motion is None:
            return
        self._apply_active_motion()

    def on_link_selected(self, link_name: str):
        if not link_name:
            return
        self.active_motion = None
        self.controller.set_target_link(link_name)
        self.controller.draw_link_frame()

    def _joint_slider_limits(self, joint_dof_idx: int):
        joint_min, joint_max = self.controller.joint_limits(joint_dof_idx)
        joint_min = self.round_float(joint_min)
        joint_max = self.round_float(joint_max)
        joint_step = self.round_float((joint_max - joint_min) / 100)
        return joint_min, joint_max, joint_step

    def launch(self):
        link_names = self.controller.link_names()
        default_link = self.controller.target_link_name
        if default_link not in link_names:
            default_link = link_names[0]

        self.controller.set_target_link(default_link)
        self.controller.draw_link_frame()

        qpos = self.robotic_arm.genesis.entity.get_qpos()[0].tolist()
        qpos = [self.round_float(v) for v in qpos]

        translation_axes = [
            ("x", "axis-x"),
            ("y", "axis-y"),
            ("z", "axis-z"),
        ]
        rotation_axes = [
            ("r", "axis-x"),
            ("p", "axis-y"),
            ("y", "axis-z"),
        ]

        with gr.Blocks(title="Sim Cartesian Controller", css=AXIS_ROW_CSS) as app:
            link_dropdown = gr.Dropdown(
                choices=link_names,
                value=default_link,
                label="Link",
                interactive=True,
            )

            gr.Markdown("**Translation**")
            with gr.Row():
                gr.Textbox(value="-", label=" ", interactive=False, scale=1)
                gr.Textbox(value="+", label=" ", interactive=False, scale=1)

            translation_buttons = []
            for axis, color_class in translation_axes:
                with gr.Row(elem_classes=[color_class]):
                    minus_btn = gr.Button(f"-{axis}", scale=1)
                    plus_btn = gr.Button(f"+{axis}", scale=1)
                    translation_buttons.append((minus_btn, plus_btn))

            gr.Markdown("**Rotation**")
            with gr.Row():
                gr.Textbox(value="-", label=" ", interactive=False, scale=1)
                gr.Textbox(value="+", label=" ", interactive=False, scale=1)

            rotation_buttons = []
            for axis, color_class in rotation_axes:
                with gr.Row(elem_classes=[color_class]):
                    minus_btn = gr.Button(f"-{axis}", scale=1)
                    plus_btn = gr.Button(f"+{axis}", scale=1)
                    rotation_buttons.append((minus_btn, plus_btn))

            joint_id_numbers = []
            joint_sliders = []
            joint_positions = []
            with gr.Row():
                gr.Textbox(value="Joint", label=" ", interactive=False, scale=2)
                gr.Textbox(value="Position", label=" ", interactive=False, scale=1)
            for joint_id, joint_name in enumerate(self.robotic_arm.joint_names):
                joint_min, joint_max, joint_step = self._joint_slider_limits(joint_id)
                joint_id_number = gr.Number(value=joint_id, visible=False)
                with gr.Row():
                    joint_slider = gr.Slider(
                        minimum=joint_min,
                        maximum=joint_max,
                        step=joint_step,
                        value=qpos[joint_id],
                        label=joint_name,
                        interactive=True,
                        scale=2,
                    )
                    joint_position = gr.Number(
                        value=qpos[joint_id],
                        label=" ",
                        interactive=False,
                        scale=1,
                    )
                joint_id_numbers.append(joint_id_number)
                joint_sliders.append(joint_slider)
                joint_positions.append(joint_position)

            def make_toggle_handler(kind, axis_id, direction):
                def on_toggle():
                    return self.toggle_motion(kind, axis_id, direction)

                return on_toggle

            for axis_id, (minus_btn, plus_btn) in enumerate(translation_buttons):
                minus_btn.click(
                    make_toggle_handler("translate", axis_id, -1),
                    outputs=joint_positions,
                )
                plus_btn.click(
                    make_toggle_handler("translate", axis_id, 1),
                    outputs=joint_positions,
                )

            for axis_id, (minus_btn, plus_btn) in enumerate(rotation_buttons):
                minus_btn.click(
                    make_toggle_handler("rotate", axis_id, -1),
                    outputs=joint_positions,
                )
                plus_btn.click(
                    make_toggle_handler("rotate", axis_id, 1),
                    outputs=joint_positions,
                )

            tick_interval = 1 / self.robotic_arm.genesis.fps
            motion_timer = gr.Timer(value=tick_interval, active=True)
            motion_timer.tick(fn=self.step_motion)

            link_dropdown.change(fn=self.on_link_selected, inputs=[link_dropdown])

            for joint_id, joint_name in enumerate(self.robotic_arm.joint_names):
                inputs = [joint_id_numbers[joint_id]] + joint_sliders
                joint_sliders[joint_id].change(
                    fn=self.set_goal_position,
                    inputs=inputs,
                    outputs=[joint_positions[joint_id]],
                )

        app.launch()

    def set_goal_position(self, joint_id, *qpos):
        joint_id = int(joint_id)

        qpos = [float(qpos_value) for qpos_value in qpos]

        self.robotic_arm.genesis.entity.control_dofs_position(qpos)
        self.robotic_arm.genesis.step()

        self.active_motion = None
        self.controller.sync_targets_from_link()

        current_pos = self.robotic_arm.genesis.entity.get_qpos()
        current_pos = current_pos[0][joint_id].item()
        current_pos = self.round_float(current_pos)

        return current_pos
