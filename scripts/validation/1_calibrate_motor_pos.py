from slobot.feetech import Feetech
import sys

# Place the robot in the position preset

if len(sys.argv) < 2:
    print("Usage: python 1_calibrate_motor_pos.py [zero|rotated|rest]")
    sys.exit(1)

preset = sys.argv[1]
Feetech.calibrate_pos(preset)

# Copy the pos array in Configuration.POS_MAP