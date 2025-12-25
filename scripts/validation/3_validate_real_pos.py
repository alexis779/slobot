from slobot.feetech import Feetech
from slobot.configuration import Configuration

import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/validation/3_validate_real_pos.py [middle|zero|rotated|rest]")
    sys.exit(1)

# Place the robot in the position preset

preset = sys.argv[1]
pos = Configuration.POS_MAP[preset]
Feetech.move_to_pos(pos)