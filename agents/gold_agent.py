"""Backward-compatibility shim for gold_agent.py.

The canonical implementation is now agents/gold.py.
This module re-exports the public API for any code that still imports
from agents.gold_agent.
"""
from agents.gold import run_gold


def execute_gold(input_files: list, sttm_path: str, run_id: str, task_description: str = "") -> list:
    """Compatibility wrapper -- delegates to agents.gold.run_gold."""
    return run_gold(input_files=input_files, sttm_path=sttm_path, run_id=run_id)
