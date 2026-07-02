"""Backward-compatibility shim for bronze_agent.py.

The canonical implementation is now agents/bronze.py.
This module re-exports the public API for any code that still imports
from agents.bronze_agent.
"""
from agents.bronze import run_bronze


def execute_bronze(input_files: list, sttm_path: str, run_id: str, task_description: str = "") -> list:
    """Compatibility wrapper -- delegates to agents.bronze.run_bronze."""
    return run_bronze(input_files=input_files, sttm_path=sttm_path, run_id=run_id)
