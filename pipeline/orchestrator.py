"""Simple Python pipeline coordinator.

NOT an LLM ReAct agent -- using an LLM supervisor was the root cause of the
I/O errors.  This module is a plain Python coordinator that calls agent
functions directly in the correct order, catches exceptions, and returns
an updated PipelineState for each phase.

Four phases, each exposed as a function called by Streamlit:

    phase1_profile_and_bronze_sttm(uploaded_files, business_intent, run_id)
        -> PipelineState
    phase2_bronze_and_silver_sttm(state)
        -> PipelineState
    phase3_silver_and_gold_sttm(state)
        -> PipelineState
    phase4_gold_and_report(state)
        -> PipelineState
"""

import uuid
import traceback
from core.logger import log
from core.audit import AuditLogger
from core.state import PipelineState


# ---------------------------------------------------------------------------
# Phase 1: Profile raw files + generate Bronze STTM
# ---------------------------------------------------------------------------

def phase1_profile_and_bronze_sttm(
    uploaded_files: list,
    business_intent: str,
    run_id: str = "",
) -> PipelineState:
    """Run the profiler and generate Bronze STTM rules.

    Called by Streamlit after the user uploads files and clicks Start.
    Returns state with profile_path and sttm_bronze_path populated.
    Human approval is required before calling phase2.

    Args:
        uploaded_files: List of saved CSV file paths in the landing zone.
        business_intent: The user's analytical question.
        run_id: Optional run ID; generated if not provided.

    Returns:
        PipelineState with status='awaiting_bronze_sttm_approval' on success
        or status='failed' with error set on failure.
    """
    if not run_id:
        run_id = str(uuid.uuid4())

    audit = AuditLogger(run_id)
    state: PipelineState = {
        "run_id": run_id,
        "status": "phase1_started",
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
        "profile_path": "",
        "sttm_bronze_path": "",
        "sttm_silver_path": "",
        "sttm_gold_path": "",
        "bronze_output_paths": [],
        "silver_output_paths": [],
        "gold_output_paths": [],
        "report_path": "",
        "error": None,
    }

    audit.log("phase1_started", "orchestrator", {
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
    })
    log(f"[ORCHESTRATOR] Phase 1 start run_id={run_id}")

    try:
        # Step 1: Profile the uploaded CSV files
        log("[ORCHESTRATOR] Running profiler")
        from agents.profiler import run_profiler
        profile_path = run_profiler(file_paths=uploaded_files, run_id=run_id)
        state["profile_path"] = profile_path
        audit.log("profiler_completed", "orchestrator", {"profile_path": profile_path})
        log(f"[ORCHESTRATOR] Profiler done -> {profile_path}")

        # Step 2: Generate Bronze STTM from the profile
        log("[ORCHESTRATOR] Generating Bronze STTM")
        from agents.sttm import run_sttm
        sttm_bronze_path = run_sttm(
            context_paths=[profile_path],
            business_intent=business_intent,
            layer="bronze",
            run_id=run_id,
        )
        state["sttm_bronze_path"] = sttm_bronze_path
        audit.log("bronze_sttm_generated", "orchestrator", {"sttm_bronze_path": sttm_bronze_path})
        log(f"[ORCHESTRATOR] Bronze STTM done -> {sttm_bronze_path}")

        state["status"] = "awaiting_bronze_sttm_approval"
        audit.log("phase1_completed", "orchestrator", {"status": "awaiting_bronze_sttm_approval"})

    except Exception as exc:
        err = f"Phase 1 failed: {exc}\n{traceback.format_exc()}"
        state["error"] = err
        state["status"] = "failed"
        audit.log("phase1_failed", "orchestrator", {"error": str(exc)})
        log(f"[ORCHESTRATOR] Phase 1 FAILED: {exc}")

    return state


# ---------------------------------------------------------------------------
# Phase 2: Execute Bronze + generate Silver STTM
# ---------------------------------------------------------------------------

def phase2_bronze_and_silver_sttm(state: PipelineState) -> PipelineState:
    """Execute approved Bronze rules and generate Silver STTM.

    Called by Streamlit after the user approves the Bronze STTM.
    Returns state with bronze_output_paths and sttm_silver_path populated.
    Human approval is required before calling phase3.

    Args:
        state: PipelineState from phase1 (must have sttm_bronze_path set).

    Returns:
        Updated PipelineState with status='awaiting_silver_sttm_approval' on success.
    """
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    state["error"] = None

    audit.log("phase2_started", "orchestrator", {"sttm_bronze_path": state.get("sttm_bronze_path")})
    log(f"[ORCHESTRATOR] Phase 2 start run_id={run_id}")

    try:
        # Step 1: Execute Bronze ingestion
        log("[ORCHESTRATOR] Running Bronze execution")
        from agents.bronze import run_bronze
        bronze_output_paths = run_bronze(
            input_files=state["uploaded_files"],
            sttm_path=state["sttm_bronze_path"],
            run_id=run_id,
        )
        state["bronze_output_paths"] = bronze_output_paths
        audit.log("bronze_executed", "orchestrator", {"bronze_output_paths": bronze_output_paths})
        log(f"[ORCHESTRATOR] Bronze execution done -> {bronze_output_paths}")

        # Step 2: Generate Silver STTM from Bronze outputs
        log("[ORCHESTRATOR] Generating Silver STTM")
        from agents.sttm import run_sttm
        sttm_silver_path = run_sttm(
            context_paths=bronze_output_paths,
            business_intent=state["business_intent"],
            layer="silver",
            run_id=run_id,
        )
        state["sttm_silver_path"] = sttm_silver_path
        audit.log("silver_sttm_generated", "orchestrator", {"sttm_silver_path": sttm_silver_path})
        log(f"[ORCHESTRATOR] Silver STTM done -> {sttm_silver_path}")

        state["status"] = "awaiting_silver_sttm_approval"
        audit.log("phase2_completed", "orchestrator", {"status": "awaiting_silver_sttm_approval"})

    except Exception as exc:
        err = f"Phase 2 failed: {exc}\n{traceback.format_exc()}"
        state["error"] = err
        state["status"] = "failed"
        audit.log("phase2_failed", "orchestrator", {"error": str(exc)})
        log(f"[ORCHESTRATOR] Phase 2 FAILED: {exc}")

    return state


# ---------------------------------------------------------------------------
# Phase 3: Execute Silver + generate Gold STTM
# ---------------------------------------------------------------------------

def phase3_silver_and_gold_sttm(state: PipelineState) -> PipelineState:
    """Execute approved Silver cleansing rules and generate Gold STTM.

    Called by Streamlit after the user approves the Silver STTM.
    Returns state with silver_output_paths and sttm_gold_path populated.
    Human approval is required before calling phase4.

    Args:
        state: PipelineState from phase2 (must have sttm_silver_path set).

    Returns:
        Updated PipelineState with status='awaiting_gold_sttm_approval' on success.
    """
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    state["error"] = None

    audit.log("phase3_started", "orchestrator", {"sttm_silver_path": state.get("sttm_silver_path")})
    log(f"[ORCHESTRATOR] Phase 3 start run_id={run_id}")

    try:
        # Step 1: Execute Silver cleansing
        log("[ORCHESTRATOR] Running Silver execution")
        from agents.silver import run_silver
        silver_output_paths = run_silver(
            input_files=state["bronze_output_paths"],
            sttm_path=state["sttm_silver_path"],
            run_id=run_id,
        )
        state["silver_output_paths"] = silver_output_paths
        audit.log("silver_executed", "orchestrator", {"silver_output_paths": silver_output_paths})
        log(f"[ORCHESTRATOR] Silver execution done -> {silver_output_paths}")

        # Step 2: Generate Gold STTM from Silver outputs (intent-driven)
        log("[ORCHESTRATOR] Generating Gold STTM")
        from agents.sttm import run_sttm
        sttm_gold_path = run_sttm(
            context_paths=silver_output_paths,
            business_intent=state["business_intent"],
            layer="gold",
            run_id=run_id,
        )
        state["sttm_gold_path"] = sttm_gold_path
        audit.log("gold_sttm_generated", "orchestrator", {"sttm_gold_path": sttm_gold_path})
        log(f"[ORCHESTRATOR] Gold STTM done -> {sttm_gold_path}")

        state["status"] = "awaiting_gold_sttm_approval"
        audit.log("phase3_completed", "orchestrator", {"status": "awaiting_gold_sttm_approval"})

    except Exception as exc:
        err = f"Phase 3 failed: {exc}\n{traceback.format_exc()}"
        state["error"] = err
        state["status"] = "failed"
        audit.log("phase3_failed", "orchestrator", {"error": str(exc)})
        log(f"[ORCHESTRATOR] Phase 3 FAILED: {exc}")

    return state


# ---------------------------------------------------------------------------
# Phase 4: Execute Gold + generate report
# ---------------------------------------------------------------------------

def phase4_gold_and_report(state: PipelineState) -> PipelineState:
    """Execute approved Gold materialisation rules and generate the HTML report.

    Called by Streamlit after the user approves the Gold STTM.
    Returns final state with gold_output_paths and report_path populated.

    Args:
        state: PipelineState from phase3 (must have sttm_gold_path set).

    Returns:
        Updated PipelineState with status='completed' on success.
    """
    run_id = state["run_id"]
    audit = AuditLogger(run_id)
    state["error"] = None

    audit.log("phase4_started", "orchestrator", {"sttm_gold_path": state.get("sttm_gold_path")})
    log(f"[ORCHESTRATOR] Phase 4 start run_id={run_id}")

    try:
        # Step 1: Execute Gold materialisation
        log("[ORCHESTRATOR] Running Gold execution")
        from agents.gold import run_gold
        gold_output_paths = run_gold(
            input_files=state["silver_output_paths"],
            sttm_path=state["sttm_gold_path"],
            run_id=run_id,
        )
        state["gold_output_paths"] = gold_output_paths
        audit.log("gold_executed", "orchestrator", {"gold_output_paths": gold_output_paths})
        log(f"[ORCHESTRATOR] Gold execution done -> {gold_output_paths}")

        # Step 2: Generate report from Gold tables
        log("[ORCHESTRATOR] Running reporter")
        from agents.reporter import run_reporter
        report_path = run_reporter(
            gold_files=gold_output_paths,
            business_intent=state["business_intent"],
            run_id=run_id,
        )
        state["report_path"] = report_path
        audit.log("report_generated", "orchestrator", {"report_path": report_path})
        log(f"[ORCHESTRATOR] Reporter done -> {report_path}")

        state["status"] = "completed"
        audit.log("phase4_completed", "orchestrator", {"status": "completed"})

    except Exception as exc:
        err = f"Phase 4 failed: {exc}\n{traceback.format_exc()}"
        state["error"] = err
        state["status"] = "failed"
        audit.log("phase4_failed", "orchestrator", {"error": str(exc)})
        log(f"[ORCHESTRATOR] Phase 4 FAILED: {exc}")

    return state
