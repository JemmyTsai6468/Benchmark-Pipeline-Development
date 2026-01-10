# src/benchmark_pipeline/config.py
"""
Configuration loader and validator for the benchmark pipeline.

This module uses Pydantic to define and validate the structure of the pipeline's
configuration, ensuring that all settings are correct before the pipeline
starts running. This adheres to the "Fail-Fast" principle.
"""
from pydantic import BaseModel, Field, DirectoryPath, FilePath
from typing import List
import yaml
from pathlib import Path

# --- Configuration Model Definitions ---

class PathsConfig(BaseModel):
    """Defines all necessary paths for the pipeline."""
    dataset_dir: Path = Field(..., description="Root directory for the structured MVTec AD dataset.")
    anomaly_maps_root: Path = Field(..., description="Root directory to save generated anomaly maps.")
    metrics_root: Path = Field(..., description="Root directory to save final evaluation metrics.")

class PipelineConfig(BaseModel):
    """
    The main configuration model for the benchmark pipeline.
    It orchestrates all other configuration parts.
    """
    device: str = Field("auto", description='Device to use, e.g., "cuda", "cpu", or "auto".')
    batch_size: int = Field(32, gt=0, description="Batch size for processing.")
    models: List[str] = Field(..., min_length=1, description="List of models to evaluate.")
    categories: List[str] = Field(..., min_length=1, description="List of MVTec AD categories to run on.")
    paths: PathsConfig

# --- Configuration Loading Function ---

def load_config(config_path: Path = Path("configs/pipeline_config.yaml")) -> PipelineConfig:
    """
    Loads, validates, and returns the pipeline configuration.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A validated PipelineConfig object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    try:
        return PipelineConfig(**config_data)
    except Exception as e:
        # Pydantic's validation error can be complex, so we wrap it.
        raise ValueError(f"Configuration validation failed: {e}")

# --- Global Configuration Object ---
# Load the configuration once and make it accessible to other modules.
# This prevents repeated file I/O and ensures consistency.
try:
    CONFIG = load_config()
except (FileNotFoundError, ValueError) as e:
    print(f"CRITICAL: Failed to load configuration. {e}")
    # Exit or handle appropriately if the config is essential for module import
    CONFIG = None
