from lerobot.motors.feetech import TorqueMode
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots import make_robot_from_config
from lerobot.motors.motors_bus import SerialMotorsBus

import lerobot.robots.so_follower.config_so_follower # register so100_follower

from slobot.configuration import Configuration
from slobot.simulation_frame import SimulationFrame
from slobot.feetech_frame import FeetechFrame

import json
import numpy as np
import time

class Feetech():
    LOGGER = Configuration.logger(__name__)
    ROBOT_TYPE = 'so100_follower'
    FOLLOWER_ID = 'follower_arm'
    LEADER_ID = 'leader_arm'

    MOTOR_MODEL = 'sts3215'
    DOFS = 6
    SYNC_READ_ATTEMPTS = 2

    PORT_FOLLOWER = '/dev/ttyACM0'
    PORT_LEADER = '/dev/ttyACM1'

    def calibrate_pos(preset):
        feetech = Feetech(qpos_map=Configuration.MJCF_QPOS_MAP)
        feetech.calibrate(preset)

    def move_to_pos(pos):
        feetech = Feetech(qpos_map=Configuration.MJCF_QPOS_MAP)
        feetech.control_position(pos)

    def __init__(self, **kwargs):
        self.qpos_map = kwargs['qpos_map']

        self.port = kwargs.get('port', Feetech.PORT_FOLLOWER)
        self.robot_id = kwargs.get('robot_id', Feetech.FOLLOWER_ID)
        self.qpos_handler = kwargs.get('qpos_handler', None)

        self.dofs = kwargs.get('dofs', Feetech.DOFS)
        self.joint_ids = range(self.dofs)

        connect = kwargs.get('connect', True)
        torque = kwargs.get('torque', True)

        self.robot = self.create_robot()
        motors_bus = kwargs.get('motors_bus', self.robot.bus)
        self.use_motors_bus(motors_bus)
        if connect:
            self.connect(torque)

    def connect(self, torque):
        self.motors_bus.connect()
        if torque:
            self.set_torque(True)

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
            for i in self.joint_ids ]

    def pos_to_qpos(self, pos):
        return [ self._steps_to_qpos(pos, id)
            for id in self.joint_ids]

    def velocity_to_qvelocity(self, velocity):
        return [ self._stepvelocity_to_velocity(velocity, i)
            for i in self.joint_ids ]

    def control_position(self, pos: list[float]):
        self._write_config('Goal_Position', pos)
        self.sync_real_to_sim(pos)

    def sync_real_to_sim(self, pos: list[float]):
        if self.qpos_handler is not None:
            feetech_frame = self.create_feetech_frame(pos)
            self.qpos_handler.handle_qpos(feetech_frame)

    def control_dofs_position(self, target_qpos):
        target_pos = self.qpos_to_pos(target_qpos)
        self.control_position(target_pos)

    def get_torque(self):
        return self._read_config('Torque_Enable')

    def set_torque(self, is_enabled: bool):
        torque_enable = TorqueMode.ENABLED.value if is_enabled else TorqueMode.DISABLED.value
        torque_enable = [
            torque_enable
            for joint_id in self.joint_ids
        ]
        self._write_config('Torque_Enable', torque_enable)

    def set_home_offset(self, home_offset):
        self._write_config("Home_Offset", home_offset)

    def set_punch(self, punch):
        self._write_config('Minimum_Startup_Force', punch)

    def set_dofs_kp(self, Kp):
        self._write_config('P_Coefficient', Kp)

    def get_dofs_kp(self):
        return self._read_config('P_Coefficient')

    def set_dofs_kv(self, Kv):
        self._write_config('D_Coefficient', Kv)

    def get_dofs_kv(self):
        return self._read_config('D_Coefficient')

    def set_dofs_ki(self, Ki):
        self._write_config('I_Coefficient', Ki)

    def get_dofs_ki(self):
        return self._read_config('I_Coefficient')

    def go_to_rest(self):
        self.go_to_preset('rest')

    def go_to_preset(self, preset):
        pos = Configuration.POS_MAP[preset]
        self.control_position(pos)
        time.sleep(1)
        self.disconnect()

    def calibrate(self, preset):
        self.set_torque(False)
        input(f"Move the arm to the {preset} position ...")
        pos = self.get_pos()
        pos_json = json.dumps(pos)
        print(f"Current position is {pos_json}")

    def create_robot(self) -> Robot:
        robot_config_class = RobotConfig.get_choice_class(Feetech.ROBOT_TYPE)
        robot_config = robot_config_class(port=self.port, id=self.robot_id)
        return make_robot_from_config(robot_config)

    def use_motors_bus(self, motors_bus: SerialMotorsBus) -> None:
        """Attach an already-connected LeRobot motors bus (avoids a second serial open)."""
        self.motors_bus = motors_bus
        self.model_resolution = motors_bus.model_resolution_table[Feetech.MOTOR_MODEL]
        self.radian_per_step = (2 * np.pi) / self.model_resolution

    def _qpos_to_steps(self, qpos, motor_index):
        steps = Configuration.MOTOR_DIRECTION[motor_index] * (qpos[motor_index] - self.qpos_map[Configuration.REFERENCE_FRAME][motor_index]) / self.radian_per_step
        return Configuration.POS_MAP[Configuration.REFERENCE_FRAME][motor_index] + int(steps)

    def _steps_to_qpos(self, pos, motor_index):
        steps = pos[motor_index] - Configuration.POS_MAP[Configuration.REFERENCE_FRAME][motor_index]
        return self.qpos_map[Configuration.REFERENCE_FRAME][motor_index] + Configuration.MOTOR_DIRECTION[motor_index] * steps * self.radian_per_step

    def _stepvelocity_to_velocity(self, step_velocity, motor_index):
        return step_velocity[motor_index] * self.radian_per_step

    def _read_config(self, key):
        motors = [
            Configuration.JOINT_NAMES[id]
            for id in self.joint_ids
        ]

        pos = self._retry_sync_read(key, motors)
        
        return [
            pos[Configuration.JOINT_NAMES[id]]
            for id in self.joint_ids
        ]

    def _retry_sync_read(self, key, motors):
        attempts = 0
        while attempts < Feetech.SYNC_READ_ATTEMPTS:
            try:
                pos = self.motors_bus.sync_read(key, motors, normalize=False)
                break
            except ConnectionError:
                attempts += 1
                if attempts == Feetech.SYNC_READ_ATTEMPTS:
                    self.motors_bus.disconnect()
                    raise
        return pos

    def _write_config(self, key, values):
        values = {
            Configuration.JOINT_NAMES[id] : values[i]
            for i, id in enumerate(self.joint_ids)
        }
        self.motors_bus.sync_write(key, values, normalize=False)

    def create_feetech_frame(self, target_pos) -> FeetechFrame:
        timestamp = time.time()
        qpos = self.pos_to_qpos(self.get_pos())
        target_qpos = self.pos_to_qpos(target_pos)
        velocity = self.get_dofs_velocity()
        force = self.get_dofs_control_force()
        return FeetechFrame(timestamp, target_qpos, qpos, velocity, force)

    def sim_positions(self, positions):
        positions = {
            joint_id+1 : positions[joint_id]
            for joint_id in range(self.dofs)
        }
        positions = self.motors_bus._unnormalize(positions)
        positions = [
            positions[joint_id+1]
            for joint_id in range(self.dofs)
        ]

        return self.pos_to_qpos(positions)
