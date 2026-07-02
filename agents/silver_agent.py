"""Backward-compatibility shim for silver_agent.py.

The canonical implementation is now agents/silver.py.
This module re-exports the public API for any code that still imports
from agents.silver_agent.
"""
from agents.silver import run_silver


def execute_silver(input_files: list, sttm_path: str, run_id: str, task_description: str = "") -> list:
    """Compatibility wrapper -- delegates to agents.silver.run_silver."""
    return run_silver(input_files=input_files, sttm_path=sttm_path, run_id=run_id)
