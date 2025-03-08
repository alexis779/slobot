from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode
from lerobot.common.robot_devices.robots.utils import make_robot_config

from slobot.configuration import Configuration

import json
import numpy as np

class Feetech():
    ROBOT_TYPE = 'so100'
    ARM_NAME = 'main'
    ARM_TYPE = 'follower'

    DOF = 6
    MODEL_RESOLUTION = 4096
    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]

    PORT = '/dev/ttyACM0'

    def calibrate_pos(preset):
        feetech = Feetech()
        feetech.calibrate(preset)

    def move_to_pos(pos):
        feetech = Feetech()
        feetech.move(pos)

    def __init__(self, **kwargs):
        self.motors_bus = self._create_motors_bus()
        self.qpos_handler = kwargs.get('qpos_handler', None)

    def disconnect(self):
        self.set_torque(False)
        self.motors_bus.disconnect()

    def get_qpos(self):
        return self.pos_to_qpos(self.get_pos())

    def get_pos(self):
        return self.motors_bus.read('Present_Position')

    def handle_qpos(self, qpos):
        pos = self.qpos_to_pos(qpos)
        self.set_pos(pos)

    def qpos_to_pos(self, qpos):
        pos = [ self._qpos_to_steps(qpos, i)
            for i in range(Feetech.DOF) ]
        return pos

    def pos_to_qpos(self, pos):
        qpos = [ self._steps_to_qpos(pos, i)
            for i in range(Feetech.DOF) ]
        return qpos

    def set_pos(self, pos):
        self.set_torque(True)
        self.motors_bus.write('Goal_Position', pos)
        if self.qpos_handler is not None:
            qpos = self.pos_to_qpos(pos)
            print("feetech qpos", qpos)
            self.qpos_handler.handle_qpos(qpos)
    
    def set_torque(self, is_enabled):
        torque_enable = TorqueMode.ENABLED.value if is_enabled else TorqueMode.DISABLED.value 
        self.motors_bus.write('Torque_Enable', torque_enable)

    def move(self, target_pos):
        self.set_pos(target_pos)
        position = self.get_pos()
        error = np.linalg.norm(target_pos - position) / Feetech.MODEL_RESOLUTION
        print("pos error=", error)

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