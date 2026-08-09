from slobot.configuration import Configuration
from slobot.feetech import Feetech
from slobot.sim_client import SimClient

# first, start server via
# uv run python sim_gradio_qpos.py

feetech = Feetech(qpos_map=Configuration.MJCF_QPOS_MAP)

url = 'http://127.0.0.1:7860' # 'https://alexis779-slobot-genesis-qpos.hf.space/'
sim_client = SimClient(url=url, step_handler=feetech)

fps = 24
sim_client.run(fps)

# Move the robot to the rest position preset
feetech.go_to_rest()