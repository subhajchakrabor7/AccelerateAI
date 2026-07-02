"""Backward-compatibility shim for sttm_generator.py.

The canonical implementation is now agents/sttm.py.
This module re-exports the public API for any code that still imports
from agents.sttm_generator.
"""
from agents.sttm import run_sttm


def generate_bronze_sttm(profile_path: str, run_id: str, task_description: str = "") -> str:
    """Compatibility wrapper -- delegates to agents.sttm.run_sttm with layer=bronze."""
    return run_sttm(
        context_paths=[profile_path],
        business_intent="",
        layer="bronze",
        run_id=run_id,
    )


def generate_silver_sttm(
    bronze_output_paths: list,
    bronze_sttm_path: str,
    run_id: str,
    task_description: str = "",
) -> str:
    """Compatibility wrapper -- delegates to agents.sttm.run_sttm with layer=silver."""
    return run_sttm(
        context_paths=bronze_output_paths,
        business_intent="",
        layer="silver",
        run_id=run_id,
    )


def generate_gold_sttm(
    silver_output_paths: list,
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str = "",
) -> str:
    """Compatibility wrapper -- delegates to agents.sttm.run_sttm with layer=gold."""
    return run_sttm(
        context_paths=silver_output_paths,
        business_intent=business_intent,
        layer="gold",
        run_id=run_id,
    )
