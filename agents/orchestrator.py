"""Backward-compatibility shim for agents/orchestrator.py.

The canonical orchestrator is now pipeline/orchestrator.py.
This module re-exports the four phase functions under the old names so any
code still importing from agents.orchestrator continues to work.
"""
from pipeline.orchestrator import (
    phase1_profile_and_bronze_sttm,
    phase2_bronze_and_silver_sttm,
    phase3_silver_and_gold_sttm,
    phase4_gold_and_report,
)

# Old name aliases used by the previous Streamlit app
run_until_bronze_sttm = phase1_profile_and_bronze_sttm
run_bronze_to_silver_sttm = phase2_bronze_and_silver_sttm
run_silver_to_gold_sttm = phase3_silver_and_gold_sttm
run_gold_and_report = phase4_gold_and_report
