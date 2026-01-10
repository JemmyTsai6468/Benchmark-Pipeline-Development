# src/benchmark_pipeline/pipeline.py
"""
The main pipeline orchestrator.

This module contains the `Pipeline` class, which is the central coordinator
of the entire benchmark process. It initializes all necessary components
and executes the pipeline steps in the correct order.
"""
import shutil

from .config import CONFIG
from .dataset_manager import DatasetManager
from .generation_runner import GenerationRunner
from .evaluation_runner import EvaluationRunner

class Pipeline:
    """
    Orchestrates the end-to-end benchmark process.
    """
    def __init__(self):
        """Initializes the pipeline, loading configuration and components."""
        if CONFIG is None:
            raise RuntimeError("Pipeline cannot be initialized because configuration failed to load.")
        
        self.config = CONFIG
        self.dataset_manager: DatasetManager = None
        self.fo_dataset: 'fo.Dataset' = None

    def run(self):
        """
        Executes the full benchmark pipeline.
        """
        print("====== Starting Benchmark Pipeline ======")
        
        try:
            # 1. Prepare directories and dataset
            self._prepare_environment()
            
            # 2. Generate anomaly maps for all model-category pairs
            self._run_all_generations()
            
            # 3. Run the evaluation scripts
            self._run_evaluation()

            print("====== Pipeline Finished Successfully! ======")

        except Exception as e:
            print(f"\n====== Pipeline Failed! ======")
            print(f"An error occurred: {e}")
            # Potentially re-raise or exit with a non-zero code
            raise

    def _prepare_environment(self):
        """
        Sets up the required directories and ensures the dataset is ready.
        """
        print("--- Preparing pipeline environment ---")
        
        # Clean and create output directories
        for path in [self.config.paths.anomaly_maps_root, self.config.paths.metrics_root]:
            if path.exists():
                print(f"Removing existing directory: {path}")
                shutil.rmtree(path)
            path.mkdir(parents=True)
            print(f"Created directory: {path}")
        
        # Ensure the dataset is downloaded and structured
        self.dataset_manager = DatasetManager(dataset_dir=self.config.paths.dataset_dir)
        self.fo_dataset = self.dataset_manager.setup_dataset()
        
        print("--- Environment preparation complete ---\n")

    def _run_all_generations(self):
        """
        Iterates through all models and categories from the config and runs
        the anomaly map generation for each.
        """
        print("--- Starting bulk anomaly map generation ---")
        for model_name in self.config.models:
            for category in self.config.categories:
                runner = GenerationRunner(
                    model_name=model_name, 
                    category=category,
                    fo_dataset=self.fo_dataset
                )
                runner.run()
        print("--- Bulk generation complete ---\n")

    def _run_evaluation(self):
        """
        Initializes and runs the evaluation process using the generated maps.
        """
        runner = EvaluationRunner(models=self.config.models)
        runner.run_evaluation()
