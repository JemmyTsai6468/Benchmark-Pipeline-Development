# src/benchmark_pipeline/evaluation_runner.py
"""
Handles the execution of the external evaluation scripts.

This module acts as a bridge between the refactored pipeline and the
original, unmodified evaluation scripts. It's responsible for creating the
necessary configuration files for them and calling them as subprocesses.
This respects the constraint of not modifying `evaluate_multiple_experiments.py`.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import List

from .config import CONFIG

class EvaluationRunner:
    """
    Manages the execution of the metric calculation and result printing scripts.
    """
    def __init__(self, models: List[str]):
        """
        Initializes the EvaluationRunner.

        Args:
            models: A list of model names that have been evaluated.
        """
        self.models = models
        self.paths = CONFIG.paths
        self.temp_eval_config_path = Path("eval_config.json")

    def run_evaluation(self):
        """
        Coordinates the entire evaluation process:
        1. Creates the temporary config for the evaluation scripts.
        2. Runs the multi-experiment evaluation script.
        3. Runs the script to print summary tables.
        """
        print("--- Starting evaluation phase ---")
        self._create_eval_config()
        self._run_multi_experiment_evaluation()
        self._print_summary()
        self._cleanup()
        print("--- Evaluation phase complete ---\n")

    def _create_eval_config(self):
        """
        Creates the `eval_config.json` file required by the
        `evaluate_multiple_experiments.py` script.
        """
        print(f"Creating temporary evaluation config at '{self.temp_eval_config_path}'...")
        
        # The evaluation script expects relative paths from the exp_base_dir
        anomaly_maps_dirs = {model_name: f"{model_name}/" for model_name in self.models}
        
        eval_config = {
            "exp_base_dir": str(self.paths.anomaly_maps_root.resolve()),
            "anomaly_maps_dirs": anomaly_maps_dirs
        }

        with open(self.temp_eval_config_path, "w") as f:
            json.dump(eval_config, f, indent=4)
        print("Temporary config created.\n")

    def _run_command(self, command: List[str], description: str, cwd: str = None):
        """A helper to run external commands and handle errors."""
        print(f"--- Running: {description} (in CWD: {cwd or '.'}) ---")
        print(f"Executing: {' '.join(command)}")
        
        # Create a copy of the current environment and update PYTHONPATH
        # This allows the subprocess to find the 'src' directory from the project root.
        env = os.environ.copy()
        project_root = Path.cwd().resolve()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{python_path}"

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=cwd,
                env=env  # Pass the modified environment
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("Stderr:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"--- Error running '{description}'! ---")
            print(f"Return code: {e.returncode}")
            print("\n--- STDOUT ---\n" + e.stdout)
            print("\n--- STDERR ---\n" + e.stderr)
            raise e
        finally:
            print(f"--- Finished: {description} ---\n")

    def _run_multi_experiment_evaluation(self):
        """
        Executes the `evaluate_multiple_experiments.py` script.
        """
        # Note: The script name is now relative to the CWD.
        command = [
            "uv", "run", "python", "evaluate_multiple_experiments.py",
            "--dataset_base_dir", str(self.paths.dataset_dir.resolve()),
            # The experiment_configs path needs to be absolute as we are changing CWD.
            "--experiment_configs", str(self.temp_eval_config_path.resolve()),
            "--output_dir", str(self.paths.metrics_root.resolve())
        ]
        self._run_command(command, "Run multi-experiment evaluation", cwd="eval_scripts")

    def _print_summary(self):
        """
        Executes the `print_metrics.py` script to display results.
        """
        # Note: The script name is now relative to the CWD.
        command = [
            "uv", "run", "python", "print_metrics.py",
            "--metrics_folder", str(self.paths.metrics_root.resolve())
        ]
        self._run_command(command, "Print summary of results", cwd="eval_scripts")
        
    def _cleanup(self):
        """Removes temporary files."""
        print("Cleaning up temporary files...")
        if self.temp_eval_config_path.exists():
            self.temp_eval_config_path.unlink()
            print(f"Removed '{self.temp_eval_config_path}'.")
