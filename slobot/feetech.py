from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import make_robot_config
from lerobot.common.motors.feetech import TorqueMode
from lerobot.common.robots import make_robot_from_config
from lerobot.common.motors import MotorsBus

from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame
from slobot.feetech_frame import FeetechFrame

import json
import numpy as np
import time

class Feetech():
    ROBOT_TYPE = 'so100_follower'
    FOLLOWER_ID = 'follower_arm'

    MODEL_RESOLUTION = 4096
    RADIAN_PER_STEP = (2 * np.pi) / MODEL_RESOLUTION
    MOTOR_DIRECTION = [-1, 1, 1, 1, 1, 1]
    JOINT_IDS = range(Configuration.DOFS)
    PORT = '/dev/ttyACM0'
    REFERENCE_FRAME = 'middle'

    def calibrate_pos(preset):
        feetech = Feetech()
        feetech.calibrate(preset)

    def move_to_pos(pos):
        feetech = Feetech()
        feetech.control_position(pos)

    def __init__(self, **kwargs):
        self.qpos_handler = kwargs.get('qpos_handler', None)
        connect = kwargs.get('connect', True)
        if connect:
            self.connect()

    def connect(self):
        self.motors_bus : MotorsBus = self._create_motors_bus()

    def disconnect(self):
        self.set_torque(False)
        self.motors_bus.disconnect()

    def get_qpos(self):
        return self.pos_to_qpos(self.get_pos())

    def get_pos(self):
        return self._read_config('Present_Position')

    def get_velocity(self):
        return self._read_config('Present_Velocity')

    def get_dofs_velocity(self):
        return self.velocity_to_qvelocity(self.get_velocity())

    def get_dofs_control_force(self):
        return self._read_config('Present_Load')
    
    def get_pos_goal(self):
        return self._read_config('Goal_Position')

    def handle_step(self, frame: SimulationFrame):
        pos = self.qpos_to_pos(frame.qpos)
        self.control_position(pos)

    def qpos_to_pos(self, qpos):
        return [ self._qpos_to_steps(qpos, i)
            for i in range(Configuration.DOFS) ]

    def pos_to_qpos(self, pos):
        return [ self._steps_to_qpos(pos, i)
            for i in range(Configuration.DOFS) ]

    def velocity_to_qvelocity(self, velocity):
        return [ self._stepvelocity_to_velocity(velocity, i)
            for i in range(Configuration.DOFS) ]

    def control_position(self, pos):
        self.set_torque(True)

        self._write_config('Goal_Position', pos, Feetech.JOINT_IDS)
        if self.qpos_handler is not None:
            feetech_frame = self.create_feetech_frame()
            self.qpos_handler.handle_qpos(feetech_frame)

    def control_dofs_position(self, target_qpos):
        target_pos = self.qpos_to_pos(target_qpos)
        self.control_position(target_pos)

    def set_torque(self, is_enabled):
        torque_enable = TorqueMode.ENABLED.value if is_enabled else TorqueMode.DISABLED.value
        torque_enable = [
            torque_enable
            for joint_id in Feetech.JOINT_IDS
        ]
        self._write_config('Torque_Enable', torque_enable, Feetech.JOINT_IDS)

    def set_punch(self, punch, ids=JOINT_IDS):
        self._write_config('Minimum_Startup_Force', punch, ids)

    def set_dofs_kp(self, Kp, ids=JOINT_IDS):
        self._write_config('P_Coefficient', Kp, ids)

    def set_dofs_kv(self, Kv, ids=JOINT_IDS):
        self._write_config('D_Coefficient', Kv, ids)

    def set_dofs_ki(self, Ki, ids=JOINT_IDS):
        self._write_config('I_Coefficient', Ki, ids)

    def go_to_rest(self):
        self.go_to_preset('rest')

    def go_to_preset(self, preset):
        pos = Configuration.POS_MAP[preset]
        self.move(pos)
        time.sleep(1)
        self.disconnect()

    def calibrate(self, preset):
        self.set_torque(False)
        input(f"Move the arm to the {preset} position ...")
        pos = self.get_pos()
        pos_json = json.dumps(pos)
        print(f"Current position is {pos_json}")

    def _create_motors_bus(self):
        robot_config = make_robot_config(Feetech.ROBOT_TYPE, port=self.PORT, id=Feetech.FOLLOWER_ID)
        robot = make_robot_from_config(robot_config)
        motors_bus = robot.bus
        motors_bus.connect()
        return motors_bus

    def _qpos_to_steps(self, qpos, motor_index):
        steps = Feetech.MOTOR_DIRECTION[motor_index] * (qpos[motor_index] - Configuration.QPOS_MAP[Feetech.REFERENCE_FRAME][motor_index]) / Feetech.RADIAN_PER_STEP
        return Configuration.POS_MAP[Feetech.REFERENCE_FRAME][motor_index] + int(steps)

    def _steps_to_qpos(self, pos, motor_index):
        steps = pos[motor_index] - Configuration.POS_MAP[Feetech.REFERENCE_FRAME][motor_index]
        return Configuration.QPOS_MAP[Feetech.REFERENCE_FRAME][motor_index] + Feetech.MOTOR_DIRECTION[motor_index] * steps * Feetech.RADIAN_PER_STEP

    def _stepvelocity_to_velocity(self, step_velocity, motor_index):
        return step_velocity[motor_index] * Feetech.RADIAN_PER_STEP

    def _read_config(self, key):
        pos = self.motors_bus.sync_read(key, normalize=False)
        return [
            pos[joint_name]
            for joint_name in Configuration.JOINT_NAMES
        ]

    def _write_config(self, key, values, ids):
        values = {
            Configuration.JOINT_NAMES[id] : values[id]
            for id in ids
        }
        self.motors_bus.sync_write(key, values, normalize=False)

    def create_feetech_frame(self) -> FeetechFrame:
        timestamp = time.time()
        qpos = self.pos_to_qpos(self.get_pos())
        velocity = self.get_dofs_velocity()
        control_force = self.get_dofs_control_force()
        return FeetechFrame(timestamp, qpos, velocity, control_force)