"""IDAMP Streamlit UI.

Guides users through the Intent-Driven Agentic Medallion Pipeline in five phases:
  1) Upload CSV files + define business intent
  2) Review and approve Bronze STTM
  3) Review and approve Silver STTM
  4) Review and approve Gold STTM
  5) View and download the executive HTML report

IMPORTANT: No top-level agent/orchestrator imports -- they are lazy-loaded only
when a button is clicked to avoid triggering LangGraph I/O at import time.
"""

import sys
import json
import html as _html_module
from pathlib import Path
from textwrap import dedent

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from core.config import LANDING_DIR, STTM_DIR

st.set_page_config(
    page_title="IDAMP - Intent-Driven Agentic Medallion Pipeline",
    page_icon="[IDAMP]",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Progress stepper constants
# ---------------------------------------------------------------------------

PROGRESS_STEPS = [
    ("upload",          "Upload and Intent"),
    ("bronze_sttm",     "Bronze STTM Review"),
    ("silver_sttm",     "Silver STTM Review"),
    ("gold_sttm",       "Gold STTM Review"),
    ("report",          "Executive Report"),
]

SELECTION_COL = "_selected_for_approval"


# ---------------------------------------------------------------------------
# Progress banner
# ---------------------------------------------------------------------------

def render_progress_banner(current_phase: str) -> None:
    phase_to_index = {phase: idx for idx, (phase, _) in enumerate(PROGRESS_STEPS)}
    current_index = phase_to_index.get(current_phase, 0)

    css = (
        "<style>"
        ".idamp-progress{display:flex;align-items:flex-start;justify-content:center;"
        "gap:0;width:100%;padding:1rem 1.25rem 0.5rem;border:1px solid rgba(120,138,160,0.24);"
        "border-radius:14px;background:linear-gradient(135deg,rgba(24,30,41,0.96),rgba(18,23,33,0.92));"
        "box-shadow:0 10px 28px rgba(0,0,0,0.22);margin-bottom:1rem;}"
        ".idamp-step{flex:1 1 0;min-width:0;display:flex;flex-direction:column;"
        "align-items:center;text-align:center;position:relative;}"
        ".idamp-step:not(:last-child)::after{content:'';position:absolute;top:1.1rem;"
        "left:calc(50% + 1.35rem);width:calc(100% - 2.7rem);height:4px;border-radius:999px;"
        "background:rgba(91,103,122,0.45);}"
        ".idamp-step.done:not(:last-child)::after{background:linear-gradient(90deg,#18b26b,#38c983);}"
        ".idamp-step.curr:not(:last-child)::after{background:linear-gradient(90deg,#f0b84b,rgba(91,103,122,0.45));}"
        ".idamp-node{width:2.7rem;height:2.7rem;border-radius:999px;display:flex;"
        "align-items:center;justify-content:center;font-size:1rem;font-weight:700;"
        "border:3px solid rgba(120,138,160,0.45);background:#18202c;color:#d6deeb;"
        "position:relative;z-index:1;box-sizing:border-box;}"
        ".idamp-step.done .idamp-node{background:linear-gradient(135deg,#159a5d,#1ec978);"
        "border-color:rgba(76,230,146,0.5);color:#fff;box-shadow:0 0 0 8px rgba(30,201,120,0.12);}"
        ".idamp-step.curr .idamp-node{background:linear-gradient(135deg,#f3b63e,#ffcf70);"
        "border-color:rgba(255,216,140,0.7);color:#1b1f27;box-shadow:0 0 0 8px rgba(243,182,62,0.16);}"
        ".idamp-label{margin-top:0.6rem;font-size:0.85rem;line-height:1.2;color:#c8d1df;"
        "font-weight:600;max-width:9rem;}"
        ".idamp-step.curr .idamp-label{color:#fff2c6;}"
        ".idamp-step.done .idamp-label{color:#dff9ea;}"
        "</style>"
    )

    parts = [css, "<div class='idamp-progress'>"]
    for idx, (_, label) in enumerate(PROGRESS_STEPS):
        if idx < current_index:
            cls, marker = "done", "+"
        elif idx == current_index:
            cls, marker = "curr", str(idx + 1)
        else:
            cls, marker = "pending", str(idx + 1)
        parts.append(
            f"<div class='idamp-step {cls}'>"
            f"<div class='idamp-node'>{marker}</div>"
            f"<div class='idamp-label'>{label}</div>"
            f"</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STTM editor helpers
# ---------------------------------------------------------------------------

def _prepare_sttm_editor(df: pd.DataFrame) -> pd.DataFrame:
    editor_df = df.copy()
    if SELECTION_COL not in editor_df.columns:
        editor_df.insert(0, SELECTION_COL, True)
    editor_df[SELECTION_COL] = editor_df[SELECTION_COL].fillna(True).astype(bool)
    return editor_df


def _extract_selected(edited_df: pd.DataFrame) -> pd.DataFrame:
    if SELECTION_COL not in edited_df.columns:
        return edited_df.copy()
    return edited_df[edited_df[SELECTION_COL]].drop(columns=[SELECTION_COL], errors="ignore")


# ---------------------------------------------------------------------------
# Audit trail helpers
# ---------------------------------------------------------------------------

def _current_audit_logs() -> list:
    run_id = st.session_state.get("run_id", "")
    if not run_id:
        return []
    from core.audit import AuditLogger
    return AuditLogger(run_id).get_logs()


def _status_css_class(entry: dict) -> str:
    status = str(entry.get("status", "")).lower()
    action = str(entry.get("action", "")).lower()
    if "fail" in status or "error" in status or "fail" in action or "error" in action:
        return "is-failed"
    if status in ("success", "completed") or "completed" in action:
        return "is-success"
    if "started" in action or "progress" in status:
        return "is-progress"
    return "is-neutral"


def _render_audit_card(entry: dict, latest: bool = False) -> str:
    ts = str(entry.get("timestamp", ""))
    time_text = ts[11:19] if len(ts) >= 19 else "--:--:--"
    agent = _html_module.escape(str(entry.get("agent", "unknown")))
    action = _html_module.escape(str(entry.get("action", "")))
    detail = _html_module.escape(str(
        entry.get("detail") or entry.get("rationale") or entry.get("error") or "OK"
    ))
    cls = _status_css_class(entry)
    badge = "<span class='latest-badge'>LATEST</span>" if latest else ""
    return (
        f"<div class='audit-card {cls}'>"
        f"<div class='audit-top'><span>{time_text}</span>{badge}</div>"
        f"<div class='audit-head'>{agent} | {action}</div>"
        f"<div class='audit-detail'>{detail}</div>"
        f"</div>"
    )


def render_audit_panel() -> None:
    st.markdown("### Audit Trail")
    logs = _current_audit_logs()
    if not logs:
        st.info("No events yet. Start the pipeline to see activity.")
        return

    css = dedent("""
    <style>
    .audit-card{border:1px solid rgba(120,138,160,0.26);border-left-width:5px;
    border-radius:10px;padding:0.5rem 0.6rem;margin-bottom:0.4rem;
    background:linear-gradient(135deg,rgba(24,30,41,0.94),rgba(16,21,31,0.92));
    box-shadow:0 6px 14px rgba(0,0,0,0.16);}
    .audit-card.is-success{border-left-color:#1ec978;}
    .audit-card.is-progress{border-left-color:#f3b63e;}
    .audit-card.is-failed{border-left-color:#f24d66;}
    .audit-card.is-neutral{border-left-color:#8ea0b8;}
    .audit-top{display:flex;justify-content:space-between;color:#9db0c7;
    font-size:0.72rem;font-weight:700;margin-bottom:0.15rem;}
    .latest-badge{background:linear-gradient(135deg,#f3b63e,#ffd27c);color:#171b24;
    border-radius:999px;padding:0.08rem 0.35rem;font-size:0.62rem;font-weight:800;}
    .audit-head{color:#e8eef8;font-size:0.82rem;font-weight:700;line-height:1.2;
    margin-bottom:0.18rem;}
    .audit-detail{color:#c8d4e5;font-size:0.76rem;line-height:1.15;
    display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;
    overflow:hidden;word-break:break-word;}
    .audit-scroll{max-height:460px;overflow-y:auto;margin-top:0.3rem;}
    </style>
    """)

    latest = logs[-1]
    previous = list(reversed(logs[:-1]))
    prev_html = "".join(_render_audit_card(e) for e in previous)
    if not prev_html:
        prev_html = "<div style='color:#9fb0c6;font-size:0.76rem;padding:0.3rem 0;'>No previous events.</div>"

    st.markdown(
        css
        + _render_audit_card(latest, latest=True)
        + f"<div class='audit-scroll'>{prev_html}</div>",
        unsafe_allow_html=True,
    )

    run_id = st.session_state.get("run_id", "run")
    st.download_button(
        "Download Audit Trail",
        data=json.dumps(logs, indent=2, ensure_ascii=True),
        file_name=f"audit_{run_id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "phase" not in st.session_state:
    st.session_state.phase = "upload"
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "run_id" not in st.session_state:
    st.session_state.run_id = ""


def _reset():
    st.session_state.phase = "upload"
    st.session_state.pipeline_state = None
    st.session_state.run_id = ""


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Intent-Driven Agentic Medallion Pipeline")
st.markdown(
    "Upload CSV files and describe your analytical goal. "
    "The pipeline transforms data through Bronze -> Silver -> Gold layers "
    "with human approval at each stage, then produces an HTML report."
)

render_progress_banner(st.session_state.phase)

main_col, audit_col = st.columns([3.2, 1.5], gap="large")

with audit_col:
    render_audit_panel()

# ---------------------------------------------------------------------------
# Main workflow area
# ---------------------------------------------------------------------------

with main_col:

    # ========================================================
    # PHASE 1: Upload and intent
    # ========================================================
    if st.session_state.phase == "upload":
        st.header("Phase 1: Upload Data and Define Intent")

        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        business_intent = st.text_area(
            "Business Intent / Question",
            placeholder="Example: What are the top-selling product categories by revenue this year?",
            height=100,
        )

        can_start = bool(uploaded_files and business_intent and business_intent.strip())

        if st.button("Start Workflow", type="primary", disabled=not can_start):
            # Save uploaded files to landing zone
            saved_paths = []
            for uf in uploaded_files:
                save_path = str(LANDING_DIR / uf.name)
                try:
                    with open(save_path, "wb") as fh:
                        fh.write(uf.getbuffer())
                    saved_paths.append(save_path)
                except Exception as exc:
                    st.error(f"Failed to save {uf.name}: {exc}")
                    st.stop()

            with st.spinner("Profiling data and generating Bronze STTM..."):
                try:
                    # Lazy import to avoid top-level LangGraph I/O
                    from pipeline.orchestrator import phase1_profile_and_bronze_sttm
                    result = phase1_profile_and_bronze_sttm(
                        uploaded_files=saved_paths,
                        business_intent=business_intent.strip(),
                    )
                    st.session_state.pipeline_state = result
                    st.session_state.run_id = result.get("run_id", "")
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.session_state.phase = "bronze_sttm"
                        st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    import traceback as _tb
                    st.code(_tb.format_exc())

    # ========================================================
    # PHASE 2: Bronze STTM review
    # ========================================================
    elif st.session_state.phase == "bronze_sttm":
        st.header("Phase 2: Review Bronze Layer STTM")
        state = st.session_state.pipeline_state

        st.info(
            "Bronze layer: raw ingestion with column standardisation and metadata injection. "
            "Review the transformation rules below. Uncheck rows to exclude them."
        )

        sttm_path = state.get("sttm_bronze_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor(df)
            st.write(f"Total rules: {len(df)}")

            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("Include", default=True),
                    "transformation_logic": st.column_config.TextColumn("Logic", width="large"),
                },
                key="bronze_sttm_editor",
                height=480,
            )

            selected_df = _extract_selected(edited_df)
            if selected_df.empty:
                st.warning("Select at least one rule to continue.")

            if st.button(
                "Approve and Continue to Silver",
                type="primary",
                use_container_width=True,
                disabled=selected_df.empty,
            ):
                selected_df.to_csv(sttm_path, index=False, encoding="utf-8")
                with st.spinner("Executing Bronze layer and generating Silver STTM..."):
                    try:
                        from pipeline.orchestrator import phase2_bronze_and_silver_sttm
                        result = phase2_bronze_and_silver_sttm(state)
                        st.session_state.pipeline_state = result
                        st.session_state.run_id = result.get("run_id", st.session_state.run_id)
                        if result.get("error"):
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.phase = "silver_sttm"
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
                        import traceback as _tb
                        st.code(_tb.format_exc())
        else:
            st.error("Bronze STTM file not found. Please restart the workflow.")
            if st.button("Restart"):
                _reset()
                st.rerun()

    # ========================================================
    # PHASE 3: Silver STTM review
    # ========================================================
    elif st.session_state.phase == "silver_sttm":
        st.header("Phase 3: Review Silver Layer STTM")
        state = st.session_state.pipeline_state

        st.info(
            "Silver layer: null handling, deduplication, type casting, date standardisation, "
            "and surrogate key injection. Review the cleansing rules below."
        )

        sttm_path = state.get("sttm_silver_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor(df)
            st.write(f"Total rules: {len(df)}")

            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("Include", default=True),
                    "transformation_logic": st.column_config.TextColumn("Logic", width="large"),
                },
                key="silver_sttm_editor",
                height=480,
            )

            selected_df = _extract_selected(edited_df)
            if selected_df.empty:
                st.warning("Select at least one rule to continue.")

            if st.button(
                "Approve and Continue to Gold",
                type="primary",
                use_container_width=True,
                disabled=selected_df.empty,
            ):
                selected_df.to_csv(sttm_path, index=False, encoding="utf-8")
                with st.spinner("Executing Silver layer and generating Gold STTM..."):
                    try:
                        from pipeline.orchestrator import phase3_silver_and_gold_sttm
                        result = phase3_silver_and_gold_sttm(state)
                        st.session_state.pipeline_state = result
                        st.session_state.run_id = result.get("run_id", st.session_state.run_id)
                        if result.get("error"):
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.phase = "gold_sttm"
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
                        import traceback as _tb
                        st.code(_tb.format_exc())
        else:
            st.error("Silver STTM file not found. Please restart the workflow.")
            if st.button("Restart"):
                _reset()
                st.rerun()

    # ========================================================
    # PHASE 4: Gold STTM review
    # ========================================================
    elif st.session_state.phase == "gold_sttm":
        st.header("Phase 4: Review Gold Layer STTM")
        state = st.session_state.pipeline_state

        st.info(
            "Gold layer: intent-driven joins, aggregations, and analytics-ready table materialisation. "
            "Review the materialisation rules below."
        )

        sttm_path = state.get("sttm_gold_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor(df)
            st.write(f"Total rules: {len(df)}")

            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("Include", default=True),
                    "transformation_logic": st.column_config.TextColumn("Logic", width="large"),
                },
                key="gold_sttm_editor",
                height=480,
            )

            selected_df = _extract_selected(edited_df)
            if selected_df.empty:
                st.warning("Select at least one rule to continue.")

            if st.button(
                "Approve and Generate Report",
                type="primary",
                use_container_width=True,
                disabled=selected_df.empty,
            ):
                selected_df.to_csv(sttm_path, index=False, encoding="utf-8")
                with st.spinner("Executing Gold layer and generating report..."):
                    try:
                        from pipeline.orchestrator import phase4_gold_and_report
                        result = phase4_gold_and_report(state)
                        st.session_state.pipeline_state = result
                        st.session_state.run_id = result.get("run_id", st.session_state.run_id)
                        if result.get("error"):
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state.phase = "report"
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
                        import traceback as _tb
                        st.code(_tb.format_exc())
        else:
            st.error("Gold STTM file not found. Please restart the workflow.")
            if st.button("Restart"):
                _reset()
                st.rerun()

    # ========================================================
    # PHASE 5: Report
    # ========================================================
    elif st.session_state.phase == "report":
        st.header("Phase 5: Executive Report")
        state = st.session_state.pipeline_state

        report_path = state.get("report_path", "")
        if report_path and Path(report_path).exists():
            with open(report_path, "r", encoding="utf-8") as fh:
                report_html = fh.read()

            st.components.v1.html(report_html, height=2200, scrolling=True)

            col1, col2 = st.columns(2)
            with col1:
                with open(report_path, "rb") as fh:
                    st.download_button(
                        label="Download HTML Report",
                        data=fh.read(),
                        file_name=f"report_{st.session_state.run_id[:8]}.html",
                        mime="text/html",
                        use_container_width=True,
                    )
            with col2:
                if st.button("Start New Analysis", use_container_width=True):
                    _reset()
                    st.rerun()
        else:
            st.error("Report file not found. Please check pipeline logs.")
            if st.button("Restart"):
                _reset()
                st.rerun()
