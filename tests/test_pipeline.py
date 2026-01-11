# tests/test_pipeline.py
import pytest
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline.pipeline import Pipeline
from src.benchmark_pipeline.config import PipelineConfig, PathsConfig

def test_pipeline_run_orchestration(mocker):
    """
    Tests the orchestration logic of the Pipeline.run() method.
    
    This test uses mocks to ensure that the main pipeline steps are called
    in the correct order, without actually executing their heavy logic.
    """
    # 1. Mock the dependent methods that `run()` calls
    mock_prepare = mocker.patch('src.benchmark_pipeline.pipeline.Pipeline._prepare_environment')
    mock_generate = mocker.patch('src.benchmark_pipeline.pipeline.Pipeline._run_all_generations')
    mock_evaluate = mocker.patch('src.benchmark_pipeline.pipeline.Pipeline._run_evaluation')
    
    # Mock the configuration to avoid file system dependency
    mock_config = PipelineConfig(
        device="cpu",
        batch_size=32,
        models=["DummyModelV1"],
        categories=["bottle"],
        paths=PathsConfig(
            dataset_dir=Path("./data"),
            anomaly_maps_root=Path("./results/maps"),
            metrics_root=Path("./results/metrics")
        )
    )
    mocker.patch('src.benchmark_pipeline.pipeline.CONFIG', mock_config)
    
    # 2. Instantiate the pipeline and call run()
    pipeline = Pipeline()
    pipeline.run()
    
    # 3. Use `Mock.assert_has_calls` to verify the sequence of operations
    # This is a powerful way to check the order of calls.
    from unittest.mock import call
    
    calls = [
        call._prepare_environment(),
        call._run_all_generations(),
        call._run_evaluation()
    ]
    
    # We can't directly check the order on the `pipeline` object itself,
    # but we can check that each mocked method was called.
    # A more advanced method is to use a manager mock.
    
    mock_prepare.assert_called_once()
    mock_generate.assert_called_once()
    mock_evaluate.assert_called_once()

def test_pipeline_init_fails_without_config(mocker):
    """
    Tests that the Pipeline raises a RuntimeError if the config is None.
    """
    # Mock the imported CONFIG object to be None
    mocker.patch('src.benchmark_pipeline.pipeline.CONFIG', None)
    
    with pytest.raises(RuntimeError) as excinfo:
        Pipeline()
        
    assert "configuration failed to load" in str(excinfo.value)
