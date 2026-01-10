# Extensible Anomaly Detection Benchmark Pipeline

## Overview

This project provides a fully automated, configurable, and extensible benchmark pipeline for evaluating anomaly detection models on standard datasets like MVTec AD.

It is designed with a clean, modern architecture to allow developers to easily add new models for evaluation and run the entire end-to-end process—from dataset setup to final metric reporting—with a single command.

## Key Features

- **🚀 Automated End-to-End Flow**: A single command executes the entire pipeline, including dataset downloading, anomaly map generation, metric calculation, and results summarization.
- **🧩 Modular Architecture**: The project follows modern software design principles (e.g., Separation of Concerns) with a clean structure:
  - `src/`: Core pipeline logic.
  - `models/`: Extensible directory for new models.
  - `eval_scripts/`: Isolated, unmodified evaluation component to ensure consistency.
- **🔌 Open-Closed Model Extensibility**: Adding a new model is as simple as creating a new file in the `models/` directory and implementing a standard interface. **No modification of the core pipeline code is required.**
- **⚙️ Configuration-Driven**: All experiments are controlled via a central, easy-to-read YAML file (`configs/pipeline_config.yaml`), where you can specify which models and dataset categories to evaluate.
- **🔒 Consistent & Reliable Evaluation**: The pipeline uses a fixed, "black box" set of evaluation scripts to ensure that all models are benchmarked under identical conditions, guaranteeing fair and reproducible results.

## Project Structure

The project is organized into the following key directories:

```
/
├─── src/benchmark_pipeline/   # The core, modern pipeline source code.
├─── scripts/run_benchmark.py  # The main and only entry point to run the pipeline.
├─── configs/                  # All user-facing configurations.
│    └─── pipeline_config.yaml
├─── models/                   # Directory for adding new, custom models.
├─── eval_scripts/             # Self-contained, legacy evaluation scripts (treated as a black box).
├─── data/                     # Default location for downloaded and structured datasets.
└─── results/                  # Default output location for anomaly maps and metrics.
```

## Setup and Installation

### Prerequisites

1.  **Python**: Version 3.12+
2.  **Package Manager**: This project uses `uv` for environment and package management.
3.  **MongoDB**: [FiftyOne](https://docs.voxel51.com/getting_started/install.html#database), the dataset management tool used in this pipeline, requires a running MongoDB instance. Please ensure you have MongoDB installed and running on `localhost:27017`.

### Installation

Install all required Python packages using `uv`:

```bash
uv pip install -r requirements.txt
```

## How to Run the Pipeline

### Step 1: Configure the Experiment

Edit `configs/pipeline_config.yaml` to define the scope of your benchmark run. Specify which models you want to evaluate and on which dataset categories.

```yaml
# configs/pipeline_config.yaml

# List of models to evaluate.
# These names must match the class names in the models/ directory.
models:
  - DummyModelV1
  - DummyModelV2
  - DummyModelV3
  # - MyNewModel # Add your custom model here

# List of MVTec AD categories to run the evaluation on.
categories:
  - bottle
  - cable
  - carpet
  # - ... and so on
```

### Step 2: Execute the Pipeline

Run the entire pipeline using the main entry point script:

```bash
uv run python scripts/run_benchmark.py
```

The script will automatically handle:
1.  Setting up the required directories.
2.  Downloading and structuring the dataset (one-time setup).
3.  Generating anomaly maps for every model-category combination.
4.  Executing the evaluation scripts.
5.  Printing a summary report to the console.

### Step 3: Check the Results

The final summary tables (AU-PRO and AU-ROC) will be printed to your console upon completion.

Detailed JSON files for each model's performance are saved in the `results/metrics/` directory, organized by model name.

## How to Add a New Model

The pipeline is designed to be easily extended with new models. Follow these three steps:

### Step 1: Create Your Model File

Create a new Python file in the `models/` directory. The file name should match the model's class name (e.g., `models/MyPatchCore.py` for a class named `MyPatchCore`).

### Step 2: Implement the `BenchmarkModel` Interface

In your new file, create a class that inherits from `src.benchmark_pipeline.model_handler.BenchmarkModel` and implements the `__init__` and `forward` methods.

**Template:**
```python
# models/MyPatchCore.py

import torch
from src.benchmark_pipeline.model_handler import BenchmarkModel

class MyPatchCore(BenchmarkModel):
    def __init__(self, image_size=(256, 256)):
        """
        Your model's initialization code goes here.
        This is where you would load weights, define layers, etc.
        """
        super().__init__(image_size)
        # Example:
        # self.patch_core_model = self.load_model_weights()

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        This method takes a batch of images and must return a batch of
        anomaly maps.

        Args:
            image_tensor: A tensor of input images, shape (N, C, H, W).

        Returns:
            A tensor of anomaly maps, shape (N, 1, H, W).
        """
        # Your model's inference logic goes here.
        # anomaly_map = self.patch_core_model(image_tensor)
        # return anomaly_map
        
        # Placeholder: returning a zero tensor
        batch_size = image_tensor.shape[0]
        return torch.zeros(batch_size, 1, self.image_size[0], self.image_size[1])

```

### Step 3: Add Model to Configuration

Finally, add the name of your new model class to the `models` list in `configs/pipeline_config.yaml`:

```yaml
models:
  - DummyModelV1
  - MyPatchCore # Your new model
```

That's it! The next time you run the pipeline, your new model will be automatically included in the benchmark.