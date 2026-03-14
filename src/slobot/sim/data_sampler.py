import torch


class DataSampler:
    """
    Infinite iterator that samples values using Sobol quasi-random sequences
    for good uniformity and low discrepancy. Reproducible via seed.
    """

    def __init__(
        self,
        ranges: list[tuple[float, float]],
        *,
        seed: int = 0,
    ):
        """
        Args:
            ranges: List of (min, max) tuples, one per dimension.
            seed: Random seed for reproducibility.
        """
        self.ranges = ranges
        self.seed = seed
        self._d = len(ranges)

    def __iter__(self):
        sampler = torch.quasirandom.SobolEngine(
            dimension=self._d, scramble=True, seed=self.seed
        )
        l_bounds = torch.tensor([r[0] for r in self.ranges])
        u_bounds = torch.tensor([r[1] for r in self.ranges])
        while True:
            raw = sampler.draw(1)
            sample = raw * (u_bounds - l_bounds) + l_bounds
            yield sample.squeeze(0)
