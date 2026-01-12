# Extensible Anomaly Detection Benchmark Pipeline

## Overview

This project provides a fully automated, configurable, and extensible benchmark pipeline for evaluating anomaly detection models on standard datasets like MVTec AD.

It is designed with a clean, modern architecture to allow developers to easily add new models for evaluation and run the entire end-to-end process—from dataset setup to final metric reporting—with a single command.

### Technologies

*   **Python**: Core language (3.12+).
*   **uv**: Fast Python package and environment management.
*   **PyTorch**: Deep learning framework for model execution and GPU acceleration.
*   **FiftyOne**: Dataset management and visualization (requires MongoDB).
*   **MongoDB**: Backend database for FiftyOne.

## Key Features

- **🚀 Automated End-to-End Flow**: A single command executes the entire pipeline, including dataset downloading, anomaly map generation, metric calculation, and results summarization.
- **🧩 Modular Architecture**: The project follows modern software design principles (e.g., Separation of Concerns) with a clean structure:
  - `src/`: Core pipeline logic.
  - `models/`: Extensible directory for new models.
  - `eval_scripts/`: Isolated, unmodified evaluation component to ensure consistency.
- **🔌 Open-Closed Model Extensibility**: Adding a new model is as simple as creating a new file in the `models/` directory and implementing a standard interface. **No modification of the core pipeline code is required.**
- **🧠 Support for Complex Models**: The pipeline architecture explicitly supports models that require a `fit()` step (e.g., for training or building a memory bank) and includes examples of advanced memory management techniques like coreset subsampling for large-scale models (see `models/PatchCore.py`).
- **⚙️ Configuration-Driven**: All experiments are controlled via a central, easy-to-read YAML file (`configs/pipeline_config.yaml`), where you can specify which models and dataset categories to evaluate.
- **⚡️ Efficient & Idiomatic Data Loading**: The pipeline uses `fiftyone.utils.torch.FiftyOneTorchDataset` with a vectorized `get_item` pattern, which is the official, recommended approach for high-performance data loading. This caches metadata in memory and transparently handles multiprocessing complexities.
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
3.  **MongoDB**: [FiftyOne](https://docs.voxel51.com/getting_started/install.html#database), the dataset management tool used in this pipeline, requires a running MongoDB instance. By default, the pipeline connects to `localhost:27017`, but you can override this in `configs/pipeline_config.yaml`.

### Installation

Install all required Python packages using `uv`:

```bash
uv pip install -r requirements.txt
```

## How to Run the Pipeline

### Step 1: Configure the Experiment

Edit `configs/pipeline_config.yaml` to define the scope of your benchmark run. Specify the computation device, database connection, batch size, which models you want to evaluate, and on which dataset categories.

```yaml
# configs/pipeline_config.yaml

# Performance settings
device: auto   # Use "cuda", "cpu", or "auto" for automatic detection.
batch_size: 32 # Number of images to process in a single batch.
num_workers: 4 # Number of worker processes for the DataLoader.

# Connection settings
database_uri: "mongodb://localhost:27017" # Connection string for the FiftyOne database.

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

The pipeline is designed to be easily extended with new models.

### Step 1: Create Your Model File

Create a new Python file in the `models/` directory. The file name should match the model's class name (e.g., `models/PatchCore.py` for a class named `PatchCore`).

### Step 2: Implement the `BenchmarkModel` Interface

In your new file, create a class that inherits from `src.benchmark_pipeline.model_handler.BenchmarkModel`. You must implement `__init__` and `forward`. You can also optionally implement `fit` if your model requires a training or fitting step.

**Template:**
```python
# models/YourNewModel.py

import torch
from src.benchmark_pipeline.model_handler import BenchmarkModel
from torch.utils.data import DataLoader

class YourNewModel(BenchmarkModel):
    def __init__(self, image_size=(256, 256)):
        """
        Your model's initialization code. Load weights, define layers, etc.
        """
        super().__init__(image_size)
        # Your layers and weights go here

    def fit(self, training_dataloader: DataLoader):
        """
        (Optional) This method is called by the pipeline before inference.
        
        Use this to train your model or build a memory bank on the provided
        'good' samples from the training set.
        
        The pipeline automatically provides a dataloader containing only
        normal training images.
        """
        print(f"[{self.__class__.__name__}] Starting fitting process...")
        # Your training/fitting logic goes here
        pass

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        This method takes a batch of test images and must return a batch
        of corresponding anomaly maps.

        Args:
            image_tensor: A tensor of input images, shape (N, C, H, W).

        Returns:
            A tensor of anomaly maps, shape (N, 1, H, W).
        """
        # Your model's inference logic goes here
        batch_size = image_tensor.shape[0]
        return torch.zeros(batch_size, 1, self.image_size[0], self.image_size[1])

```

### Step 3: Add Model to Configuration

Finally, add the name of your new model class to the `models` list in `configs/pipeline_config.yaml`:

```yaml
models:
  - PatchCore
  - YourNewModel # Your new model
```

That's it! The pipeline will automatically call `fit()` if it exists and then `forward()` for evaluation.

### Advanced: Handling Memory-Intensive Models (e.g., PatchCore)

Some models, like PatchCore, create a large memory bank of features from the training set. This can easily cause a **CUDA Out of Memory (OOM)** error during inference, as the distance calculation between all test patches and all memory bank patches becomes too large.

This project's `PatchCore` implementation demonstrates a standard solution: **Coreset Subsampling**.

The idea is to select a smaller, representative subset of the full memory bank. This drastically reduces memory usage and computation time during inference, often with a minimal impact on accuracy.

**How it's implemented in `models/PatchCore.py`:**

1.  **Configurable Ratio**: The `PatchCore` `__init__` method accepts a `coreset_sampling_ratio` (defaulting to `0.01` or 1%).

    ```python
    class PatchCore(BenchmarkModel):
        def __init__(self, image_size=(256, 256), coreset_sampling_ratio: float = 0.01):
            super().__init__(image_size)
            self.coreset_sampling_ratio = coreset_sampling_ratio
            # ...
    ```

2.  **Subsampling in `fit()`**: After the full memory bank is built, it is randomly subsampled to the desired size before being stored.

    ```python
    # In the fit() method...
    full_memory_bank = torch.cat(all_features, dim=0)

    # Coreset Subsampling
    if self.coreset_sampling_ratio < 1.0:
        num_features_to_keep = int(full_memory_bank.shape[0] * self.coreset_sampling_ratio)
        perm = torch.randperm(full_memory_bank.shape[0])
        self.memory_bank = full_memory_bank[perm[:num_features_to_keep]]
    ```

This pattern is a best practice for implementing similar memory-intensive models in this pipeline. By tuning the sampling ratio, you can effectively manage the trade-off between performance and resource consumption.

---

## Development Notes

### Unit Testing

To ensure the reliability and maintainability of the pipeline, unit tests have been implemented using `pytest` and `pytest-mock`. These tests focus on verifying the correctness of individual components in isolation, making refactoring safer and identifying bugs earlier in the development cycle.

**Running Tests:**

To run the entire test suite, ensure your development environment is active (e.g., via `uv pip install -r requirements.txt`) and execute the following command from the project root:

```bash
uv run pytest tests/
```

**Current Coverage:**

Unit tests currently cover:
*   **`ModelHandler`**: Verifies the dynamic loading and instantiation of models.
*   **`utils.py`**: Ensures the correct behavior of utility functions, including device selection and error handling for invalid inputs.
*   **`Pipeline` Orchestration**: Confirms that the main pipeline steps are called in the correct order during execution, using mocks to isolate the orchestration logic.

### High-Performance Data Loading with PyTorch

This project uses the officially recommended `fiftyone.utils.torch.FiftyOneTorchDataset` to interface between the FiftyOne dataset and the PyTorch `DataLoader`.

The data loading is highly optimized using a **vectorized** strategy. In `src/benchmark_pipeline/generation_runner.py`, you will see:
1.  A custom `GetItem` class is defined to specify which data fields are needed (e.g., `filepath`, `defect`).
2.  The `FiftyOneTorchDataset` is initialized with `vectorize=True`.

This approach loads all required sample metadata from the database into memory at once, avoiding costly per-sample database queries inside the data loading loop. This is the most performant and robust way to handle data loading from FiftyOne in a multi-worker PyTorch environment, as it transparently deals with database connection safety across processes.