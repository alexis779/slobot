#!/usr/bin/env python3
"""Teleoperation script - control a follower arm with a leader arm.

Uses asynchronous multi-process architecture with Linux FIFO queues.
"""

import argparse

from slobot.feetech import Feetech
from slobot.teleop.asyncprocessing import AsyncTeleoperator

parser = argparse.ArgumentParser(description="Teleoperate a robot arm")
parser.add_argument('--fps', type=int, default=30, help="Frames per second")
parser.add_argument('--recording-id', type=str, help="Rerun recording ID")
parser.add_argument('--leader-port', default=Feetech.PORT1, help="Leader arm serial port")
parser.add_argument('--follower-port', default=Feetech.PORT0, help="Follower arm serial port")
parser.add_argument('--camera-id', type=int, default=None, help="Webcam device ID (enables webcam if set)")
parser.add_argument('--width', type=int, default=640, help="Camera/sim width")
parser.add_argument('--height', type=int, default=480, help="Camera/sim height")
parser.add_argument('--sim', action="store_true", default=True, help="Enable simulation")

args = parser.parse_args()

teleoperator = AsyncTeleoperator(**vars(args))
teleoperator.run()
