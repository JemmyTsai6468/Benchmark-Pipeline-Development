# generic_util.py
"""
This is a compatibility shim to support the legacy evaluation scripts.

The core logic has been moved to `src/benchmark_pipeline/utils.py`.
This file simply re-exports the contents of the new module, so that
old scripts that rely on `import generic_util` continue to work without
modification. This avoids code duplication while maintaining backward
compatibility.
"""
from src.benchmark_pipeline.utils import *
