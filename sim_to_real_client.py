from slobot.feetech import Feetech
from slobot.sim_client import SimClient

# first, start server via
# python sim_gradio_qpos.py

feetech = Feetech()

sim_client = SimClient(step_handler=feetech)

fps = 24
sim_client.run(fps)

# Move the robot to the rest position preset
feetech.go_to_rest()
