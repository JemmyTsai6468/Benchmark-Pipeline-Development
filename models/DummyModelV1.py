# models/DummyModelV1.py
"""A dummy model that returns a random noise anomaly map."""
import torch
from src.benchmark_pipeline.model_handler import BenchmarkModel

class DummyModelV1(BenchmarkModel):
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Returns a random anomaly map."""
        batch_size = image_tensor.shape[0]
        return torch.rand(batch_size, 1, self.image_size[0], self.image_size[1])
