# tests/test_model_handler.py
import pytest
import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow for absolute imports
# This is necessary because pytest runs from the root directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline.model_handler import ModelHandler, BenchmarkModel

# To test the failure case of a model not inheriting from BenchmarkModel,
# we need to create a dummy file for it. We'll do this within the tests/ dir.
def setup_module(module):
    """Create a temporary 'bad' model file for testing purposes."""
    bad_model_code = """
class BadModel:  # Does not inherit from BenchmarkModel
    def __init__(self, image_size=(256, 256)):
        pass
    def forward(self, image_tensor):
        return image_tensor
"""
    with open("tests/BadModel.py", "w") as f:
        f.write(bad_model_code)
        
    # Also create a dummy good model for isolated testing
    good_model_code = """
from src.benchmark_pipeline.model_handler import BenchmarkModel
import torch

class GoodModel(BenchmarkModel):
    def __init__(self, image_size=(256, 256)):
        super().__init__(image_size)
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image_tensor)
"""
    with open("tests/GoodModel.py", "w") as f:
        f.write(good_model_code)

def teardown_module(module):
    """Remove the temporary model file."""
    Path("tests/BadModel.py").unlink()
    Path("tests/GoodModel.py").unlink()
    # Also remove __pycache__ if it exists
    pycache = Path("tests/__pycache__")
    if pycache.exists():
        import shutil
        shutil.rmtree(pycache)


def test_model_handler_success():
    """
    Tests if the ModelHandler can successfully load a valid, existing model.
    We use a dummy model within the `models` directory for this.
    """
    # Assuming 'DummyModelV1' exists and is valid.
    handler = ModelHandler("DummyModelV1")
    model_instance = handler.get_model_instance()
    assert model_instance is not None
    assert isinstance(model_instance, BenchmarkModel)
    assert model_instance.training is False # Should be in eval mode by default

def test_model_handler_model_not_found():
    """
    Tests if ModelHandler raises an ImportError for a non-existent model.
    """
    with pytest.raises(ImportError) as excinfo:
        ModelHandler("ThisModelDoesNotExist")
    # Check if the error message is informative
    assert "Could not find or import model 'ThisModelDoesNotExist'" in str(excinfo.value)

def test_model_handler_wrong_base_class():
    """
    Tests if ModelHandler raises a TypeError for a model that does not
    inherit from the BenchmarkModel base class.
    """
    # We test this by pointing the handler to our temporary 'tests' module
    with pytest.raises(TypeError) as excinfo:
        ModelHandler("BadModel", models_module="tests")
    assert "must inherit from BenchmarkModel" in str(excinfo.value)

def test_get_model_instance_returns_correct_type():
    """
    Tests if the get_model_instance method returns a correct instance.
    """
    handler = ModelHandler("GoodModel", models_module="tests")
    instance = handler.get_model_instance()
    assert isinstance(instance, BenchmarkModel)
    # The class name should be GoodModel
    assert instance.__class__.__name__ == "GoodModel"

