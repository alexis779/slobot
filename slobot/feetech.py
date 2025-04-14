from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode
from lerobot.common.robot_devices.robots.utils import make_robot_config

from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame
from slobot.feetech_frame import FeetechFrame

import json
import numpy as np
import time

class Feetech():
    ROBOT_TYPE = 'so100'
    ARM_NAME = 'main'
    ARM_TYPE = 'follower'

    MODEL_RESOLUTION = 4096
    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]
    JOINT_IDS = [0, 1, 2, 3, 4, 5]
    PORT = '/dev/ttyACM0'

    def calibrate_pos(preset):
        feetech = Feetech()
        feetech.calibrate(preset)

    def move_to_pos(pos):
        feetech = Feetech()
        feetech.move(pos)

    def __init__(self, **kwargs):
        self.qpos_handler = kwargs.get('qpos_handler', None)

    def connect(self):
        self.motors_bus = self._create_motors_bus()

    def disconnect(self):
        self.set_torque(False)
        self.motors_bus.disconnect()

    def get_qpos(self):
        return self.pos_to_qpos(self.get_pos())

    def get_pos(self):
        return self.motors_bus.read('Present_Position')

    def get_dofs_control_force(self):
        return self.motors_bus.read('Present_Load')
    
    def get_pos_goal(self):
        return self.motors_bus.read('Goal_Position')

    def handle_step(self, frame: SimulationFrame):
        pos = self.qpos_to_pos(frame.qpos)
        self.control_position(pos)

    def qpos_to_pos(self, qpos):
        pos = [ self._qpos_to_steps(qpos, i)
            for i in range(Configuration.DOFS) ]
        return pos

    def pos_to_qpos(self, pos):
        qpos = [ self._steps_to_qpos(pos, i)
            for i in range(Configuration.DOFS) ]
        return qpos

    def control_position(self, pos):
        self.set_torque(True)
        self.motors_bus.write('Goal_Position', pos)
        if self.qpos_handler is not None:
            feetech_frame = self.create_feetech_frame()
            self.qpos_handler.handle_qpos(feetech_frame)

    def control_dofs_position(self, target_qpos):
        target_pos = self.qpos_to_pos(target_qpos)
        self.control_position(target_pos)

    def set_torque(self, is_enabled):
        torque_enable = TorqueMode.ENABLED.value if is_enabled else TorqueMode.DISABLED.value
        self._write_config('Torque_Enable', torque_enable)

    def set_dofs_kp(self, Kp, ids=JOINT_IDS):
        self._write_config('P_Coefficient', Kp, ids)

    def set_dofs_kv(self, Kv, ids=JOINT_IDS):
        self._write_config('D_Coefficient', Kv, ids)

    def move(self, target_pos):
        self.control_position(target_pos)
        position = self.get_pos()
        error = np.linalg.norm(target_pos - position) / Feetech.MODEL_RESOLUTION
        print("pos error=", error)

    def go_to_rest(self):
        self.go_to_preset('rest')

    def go_to_preset(self, preset):
        pos = Configuration.POS_MAP[preset]
        self.move(pos)
        time.sleep(1)
        self.disconnect()

    def calibrate(self, preset):
        input(f"Move the arm to the {preset} position ...")
        pos = self.get_pos()
        pos_json = json.dumps(pos.tolist())
        print(f"Current position is {pos_json}")

    def _create_motors_bus(self):
        self.robot_config = make_robot_config(Feetech.ROBOT_TYPE)
        motors = self.robot_config.follower_arms[Feetech.ARM_NAME].motors
        config = FeetechMotorsBusConfig(port=self.PORT, motors=motors)
        motors_bus = FeetechMotorsBus(config)
        motors_bus.connect()
        return motors_bus

    def _qpos_to_steps(self, qpos, motor_index):
        steps = Feetech.MOTOR_DIRECTION[motor_index] * Feetech.MODEL_RESOLUTION * (qpos[motor_index] - Configuration.QPOS_MAP['rotated'][motor_index]) / (2 * np.pi)
        return Configuration.POS_MAP['rotated'][motor_index] + int(steps)

    def _steps_to_qpos(self, pos, motor_index):
        steps = pos[motor_index] - Configuration.POS_MAP['rotated'][motor_index]
        return Configuration.QPOS_MAP['rotated'][motor_index] + Feetech.MOTOR_DIRECTION[motor_index] * steps * (2 * np.pi) / Feetech.MODEL_RESOLUTION

    def _write_config(self, key, values, ids=JOINT_IDS):
        motor_names = self._motor_names(ids)
        self.motors_bus.write(key, values, motor_names)

    def _motor_names(self, ids):
        return [
            self._motor_name(id)
            for id in ids
        ]

    def _motor_name(self, id):
        return Configuration.JOINT_NAMES[id]

    def create_feetech_frame(self) -> FeetechFrame:
        timestamp = time.time()
        qpos_current = self.pos_to_qpos(self.get_pos())
        qpos_goal = self.pos_to_qpos(self.get_pos_goal())
        force = self.get_dofs_control_force()
        return FeetechFrame(timestamp, qpos_current, qpos_goal, force)