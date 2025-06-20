from gradio_client import Client
from slobot.simulation_frame import SimulationFrame
from slobot.configuration import Configuration

import time

class SimClient():

    LOGGER = Configuration.logger(__name__)

    def __init__(self, **kwargs):
        url = kwargs['url']
        self.client = Client(url)
        self.step_handler = kwargs['step_handler']

    def run(self, fps):
        job = self.client.submit(fps=fps, api_name="/sim_qpos")
        previous_time = time.time()
        period = 1.0 / fps
        for qpos in job:
            SimClient.LOGGER.info(f"Received qpos {qpos}")
            simulation_frame = SimulationFrame(0, qpos)

            current_time = time.time()
            delta = current_time - (previous_time + period)
            if delta < 0:
                time.sleep(-delta)

            self.step_handler.handle_step(simulation_frame)
            previous_time = max(current_time, previous_time + period)
