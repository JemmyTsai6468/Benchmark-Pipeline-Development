# tests/test_utils.py
import pytest
import torch
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline.utils import get_device

def test_get_device_cpu():
    """Tests if the function correctly returns a CPU device."""
    device = get_device("cpu")
    assert device.type == "cpu"

def test_get_device_cuda_explicitly():
    """Tests if the function correctly returns a CUDA device when specified."""
    # This test will only run if CUDA is actually available.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test_get_device_cuda_explicitly")
    
    device = get_device("cuda")
    assert device.type == "cuda"

def test_get_device_auto_cuda_available(mocker):
    """
    Tests the 'auto' option when CUDA is available.
    We use a mock to force the condition.
    """
    mocker.patch('torch.cuda.is_available', return_value=True)
    device = get_device("auto")
    assert device.type == "cuda"

def test_get_device_auto_cuda_not_available(mocker):
    """
    Tests the 'auto' option when CUDA is NOT available.
    We use a mock to force the condition.
    """
    mocker.patch('torch.cuda.is_available', return_value=False)
    device = get_device("auto")
    assert device.type == "cpu"

def test_get_device_invalid_input():
    """Tests if the function raises an error for invalid device strings."""
    with pytest.raises(ValueError) as excinfo:
        get_device("invalid_device")
    assert "Invalid device specified" in str(excinfo.value)
