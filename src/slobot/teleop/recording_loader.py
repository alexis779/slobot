from functools import cached_property

import pyarrow as pa
import rerun as rr
import torch

from slobot.configuration import Configuration
from slobot.metrics.rerun_metrics import RerunMetrics

class RecordingLoader:

    def __init__(self, rrd_file: str):
        self.rrd_file = rrd_file
        self.arrow_table = self.load_arrow_table()

    def load_arrow_table(self) -> pa.Table:
        with rr.server.Server(datasets={RerunMetrics.APPLICATION_ID: [self.rrd_file]}) as server:
            client = server.client()
            dataset = client.get_dataset(RerunMetrics.APPLICATION_ID)
            df = dataset.reader(index=RerunMetrics.TIME_METRIC)
            record_batches = df.collect()
            return pa.Table.from_batches(record_batches)

    @cached_property
    def timestamp(self) -> torch.Tensor:
        column = self.arrow_table.column('log_time')
        # Convert timestamps to float seconds
        timestamps = [ts.as_py().timestamp() for ts in column]
        return torch.tensor(timestamps, dtype=torch.float64)

    @cached_property
    def action(self) -> torch.Tensor:
        """Returns tensor of shape (num_frames, num_dofs)"""
        return self.get_metric_tensor(RerunMetrics.CONTROL_POS_METRIC)

    @cached_property
    def observation_state(self) -> torch.Tensor:
        """Returns tensor of shape (num_frames, num_dofs)"""
        return self.get_metric_tensor(RerunMetrics.REAL_QPOS_METRIC)

    def frame_action(self, frame_id: int) -> torch.Tensor:
        """Returns action tensor of shape (num_dofs,) at the specified frame_id"""
        return self.action[frame_id]

    def frame_observation_state(self, frame_id: int) -> torch.Tensor:
        """Returns observation state tensor of shape (num_dofs,) at the specified frame_id"""
        return self.observation_state[frame_id]

    def get_metric_tensor(self, metric_name: str) -> torch.Tensor:
        """Returns tensor of shape (num_frames, num_dofs)"""
        columns = [
            self.get_scalar_column(f"{metric_name}/{joint_name}:Scalars:scalars")
            for joint_name in Configuration.JOINT_NAMES
        ]
        return torch.stack(columns, dim=1)

    def get_scalar_column(self, column_name: str) -> torch.Tensor:
        column = self.arrow_table.column(column_name)
        # Each cell is a list with one scalar value; replace nulls with NaN
        values = [cell.as_py()[0] if cell.as_py() else float('nan') for cell in column]
        tensor = torch.tensor(values, dtype=torch.float32)
        return self.forward_fill(tensor)

    def forward_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with the previous valid value."""
        mask = torch.isnan(tensor)
        if not mask.any():
            return tensor
        idx = torch.where(~mask, torch.arange(len(tensor)), 0)
        idx = torch.cummax(idx, dim=0).values
        return tensor[idx]
