# models/DummyModelV3.py
"""A dummy model that returns a vertical gradient anomaly map."""
import torch
from src.benchmark_pipeline.model_handler import BenchmarkModel

class DummyModelV3(BenchmarkModel):
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Returns a vertical gradient anomaly map."""
        batch_size = image_tensor.shape[0]
        gradient = torch.linspace(0, 1, self.image_size[0]).unsqueeze(1).repeat(1, self.image_size[1])
        return gradient.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
