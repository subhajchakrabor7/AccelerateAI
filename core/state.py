"""Pipeline state TypedDict shared across all pipeline phases and the UI."""

from __future__ import annotations
from typing import Optional, TypedDict


class PipelineState(TypedDict):
    """Shared mutable state passed through the orchestrator phases.

    Keys read by Streamlit UI must not be renamed without updating streamlit_app.py.
    """
    run_id: str
    status: str
    uploaded_files: list
    business_intent: str
    profile_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_output_paths: list
    silver_output_paths: list
    gold_output_paths: list
    report_path: str
    error: Optional[str]
