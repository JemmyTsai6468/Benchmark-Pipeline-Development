# src/benchmark_pipeline/model_handler.py
"""
Defines the model interface and a handler for loading models.

This module is central to making the pipeline extensible. It defines an
abstract base class (`BenchmarkModel`) that all models must implement. The
`ModelHandler` class can then dynamically load any model that conforms to
this interface, adhering to the Open-Closed Principle.
"""
import importlib
from abc import ABC, abstractmethod
import torch
from typing import Type

class BenchmarkModel(ABC, torch.nn.Module):
    """
    Abstract Base Class for all benchmark models.
    
    Any model intended for evaluation by this pipeline must inherit from this
    class and implement the `forward` method. It already inherits from
    `torch.nn.Module` to ensure compatibility with PyTorch-based models.
    """
    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.image_size = image_size

    @abstractmethod
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes an input image tensor and returns an anomaly map.

        Args:
            image_tensor: A batch of images, typically of shape
                          (N, C, H, W).

        Returns:
            A tensor representing the anomaly map, typically of shape
            (N, 1, H, W).
        """
        pass

class ModelHandler:
    """
    Handles the dynamic loading of benchmark models.
    """
    def __init__(self, model_name: str, models_module: str = "models"):
        """
        Initializes the ModelHandler.

        Args:
            model_name: The name of the model to load (e.g., "DummyModelV1").
                        This name must correspond to a class of the same name.
            models_module: The name of the Python module/package where the model
                           classes are defined (e.g., "models").
        """
        self.model_name = model_name
        self.models_module = models_module
        self._model_class = self._find_model_class()

    def _find_model_class(self) -> Type[BenchmarkModel]:
        """
        Dynamically imports and retrieves the model class from the models module.

        Returns:
            The model's class type.

        Raises:
            ImportError: If the model class cannot be found.
        """
        try:
            module = importlib.import_module(f"{self.models_module}.{self.model_name}")
            model_class = getattr(module, self.model_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not find or import model '{self.model_name}'. "
                f"Ensure there is a file named '{self.models_module}/{self.model_name}.py' "
                f"containing a class named '{self.model_name}'."
            ) from e
        
        if not issubclass(model_class, BenchmarkModel):
            raise TypeError(f"Model class '{self.model_name}' must inherit from BenchmarkModel.")
            
        return model_class

    def get_model_instance(self, image_size=(256, 256)) -> BenchmarkModel:
        """
        Creates and returns an instance of the loaded model.

        Args:
            image_size: The image size the model should be configured for.

        Returns:
            An instance of the benchmark model.
        """
        print(f"Instantiating model: {self.model_name}")
        model = self._model_class(image_size=image_size)
        model.eval()  # Set to evaluation mode by default
        return model
