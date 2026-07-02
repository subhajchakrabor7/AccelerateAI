"""Data profiling agent using LangGraph create_react_agent.

Public API:
    run_profiler(file_paths, run_id) -> str  (returns profile JSON path)
"""

import io
import contextlib
import json
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from core.config import (
    PROFILES_DIR, LLM_PROVIDER,
    GROQ_API_KEY, GROQ_MODEL,
    GOOGLE_API_KEY, GEMINI_MODEL,
)
from core.audit import AuditLogger
from core.logger import log


# ---------------------------------------------------------------------------
# stdout/stderr safety wrapper -- must wrap ALL langgraph calls
# ---------------------------------------------------------------------------

def _invoke_agent(agent, inputs: dict, max_retries: int = 4) -> dict:
    """Invoke a LangGraph agent with stdout/stderr redirected and rate-limit retry."""
    for attempt in range(max_retries):
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                return agent.invoke(inputs)
        except Exception as exc:
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                wait = 15 * (attempt + 1)
                log(f"[PROFILER] Rate limit hit, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return agent.invoke(inputs)
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return agent.invoke(inputs)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Pure Python helpers (no LLM)
# ---------------------------------------------------------------------------

def _inspect_files(file_paths: list) -> dict:
    """Lightweight CSV preview: shape, columns, dtypes, 3 sample values."""
    summary = {}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            summary[os.path.basename(fp)] = {"error": str(exc)}
            continue
        name = os.path.splitext(os.path.basename(fp))[0]
        col_previews = {}
        for col in df.columns:
            col_previews[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist(),
                "null_count": int(df[col].isnull().sum()),
            }
        summary[name] = {
            "file": fp,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_preview": col_previews,
        }
    return summary


def _compute_stats(file_paths: list) -> dict:
    """Full column-level statistics across all CSV files."""
    combined: dict = {"files": [], "datasets": {}}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            log(f"[PROFILER] Could not read {fp}: {exc}")
            continue
        name = os.path.splitext(os.path.basename(fp))[0]
        combined["files"].append(fp)
        ds: dict = {
            "file": fp,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": {},
        }
        for col in df.columns:
            info: dict = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            else:
                info["sample_values"] = df[col].dropna().head(5).tolist()
            ds["columns"][col] = info
        combined["datasets"][name] = ds
    return combined


# ---------------------------------------------------------------------------
# Tool factory -- tools are closures over file_paths
# ---------------------------------------------------------------------------

def _make_tools(file_paths: list, run_id: str):
    @tool
    def inspect_files(confirmation: str = "execute") -> str:
        """Preview each uploaded CSV file to understand structure before full profiling.

        Returns a JSON summary with shape, column names, dtypes, and 3 sample values
        per column. Call this FIRST to understand the data before running full stats.
        """
        return json.dumps(_inspect_files(file_paths), default=str)

    @tool
    def profile_files(confirmation: str = "execute") -> str:
        """Compute full column-level statistics for all uploaded CSV files.

        Reads each CSV and computes dtype, null count, null pct, unique count,
        sample values, and numeric distributions (min/max/mean). Returns a JSON
        statistics object. Call this AFTER inspect_files.
        """
        stats = _compute_stats(file_paths)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_filename = f"profile_combined_{ts}.json"
        profile_path = str(PROFILES_DIR / profile_filename)
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        log(f"[PROFILER] Profile saved -> {profile_path}")
        return json.dumps({"profile_path": profile_path, "stats": stats}, default=str)

    return [inspect_files, profile_files]


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a data profiling specialist for a Medallion pipeline. "
    "Your job is to inspect uploaded CSV files and produce a full data profile. "
    "\n\nFollow these steps exactly:"
    "\n1. Call inspect_files to see the column names, dtypes, and sample values."
    "\n2. Call profile_files to compute full statistics and save the profile JSON."
    "\n3. Return the profile_path from the profile_files result as your final answer."
    "\n\nReturn ONLY the profile_path string as your final message. No other text."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_profiler(file_paths: list, run_id: str) -> str:
    """Profile CSV files and return the path to the saved profile JSON.

    Args:
        file_paths: List of CSV file paths to profile.
        run_id: Unique identifier for this pipeline run.

    Returns:
        str: Absolute path to the saved combined profile JSON file.
    """
    audit = AuditLogger(run_id)
    audit.log("started", "profiler", {"input_files": file_paths})
    log(f"[PROFILER] Starting run_id={run_id} files={file_paths}")

    tools = _make_tools(file_paths, run_id)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        llm = _make_llm()
        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    try:
        result = _invoke_agent(agent, {"messages": [HumanMessage(content=(
            f"Profile these CSV files for run_id={run_id}: {file_paths}. "
            "Inspect them first, then compute full statistics and save the profile."
        ))]})
    except Exception as exc:
        audit.log("failed", "profiler", {"error": str(exc)})
        log(f"[PROFILER] Agent failed: {exc}")
        # Fallback: compute stats directly without LLM
        stats = _compute_stats(file_paths)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = str(PROFILES_DIR / f"profile_combined_{ts}.json")
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        audit.log("completed_fallback", "profiler", {"profile_path": profile_path})
        return profile_path

    # Extract profile_path from agent message history
    profile_path = ""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        # Try to parse as JSON first (tool result)
        text = content
        for fence in ("```json", "```"):
            if fence in text:
                parts = text.split(fence)
                if len(parts) >= 3:
                    text = parts[1]
                    break
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "profile_path" in parsed:
                profile_path = parsed["profile_path"]
                break
        except (json.JSONDecodeError, ValueError):
            pass
        # Look for a bare path string ending in .json
        if ".json" in content and "profile_combined" in content:
            for word in content.split():
                word = word.strip('",\'')
                if "profile_combined" in word and word.endswith(".json"):
                    profile_path = word
                    break
        if profile_path:
            break

    # Fallback: recompute if we couldn't extract the path
    if not profile_path or not Path(profile_path).exists():
        log("[PROFILER] Could not extract profile_path from messages, recomputing")
        stats = _compute_stats(file_paths)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = str(PROFILES_DIR / f"profile_combined_{ts}.json")
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)

    audit.log("completed", "profiler", {"profile_path": profile_path})
    log(f"[PROFILER] Done -> {profile_path}")
    return profile_path
