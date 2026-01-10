# scripts/run_benchmark.py
"""
The main entry point for the user to execute the benchmark pipeline.

This script is kept intentionally simple. Its only job is to import the
main Pipeline class, instantiate it, and call its `run` method. This
provides a clean and stable entry point for the entire application.
"""
import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow for absolute imports
# This is a common pattern for making script execution independent of the
# current working directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import fiftyone as fo
from src.benchmark_pipeline.pipeline import Pipeline

# --- Configure FiftyOne to use a local database ---
# This must be done before any other fiftyone operations.
fo.config.database_uri = "mongodb://localhost:27017"

def main():
    """
    Initializes and runs the benchmark pipeline.
    """
    try:
        pipeline = Pipeline()
        pipeline.run()
    except Exception as e:
        # The pipeline itself handles detailed error logging.
        # This is a final catch-all for any unexpected failures during init.
        print(f"A critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
