# models/DummyModelV2.py
"""A dummy model that returns a constant value anomaly map."""
import torch
from src.benchmark_pipeline.model_handler import BenchmarkModel

class DummyModelV2(BenchmarkModel):
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Returns a constant anomaly map."""
        batch_size = image_tensor.shape[0]
        return torch.full((batch_size, 1, self.image_size[0], self.image_size[1]), 0.5)
