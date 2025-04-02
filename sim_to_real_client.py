from slobot.feetech import Feetech
from slobot.sim_client import SimClient
from slobot.configuration import Configuration

# first, start server via
# python sim_gradio_qpos.py

feetech = Feetech()

sim_client = SimClient(step_handler=feetech)

res = Configuration.LD
fps = 24
sim_client.run(res, fps)

# Move the robot to the rest position preset
feetech.go_to_rest()
