"""STTM (Source-to-Target Mapping) generation agent using LangGraph create_react_agent.

Public API:
    run_sttm(context_paths, business_intent, layer, run_id) -> str  (STTM CSV path)
"""

import io
import contextlib
import json
import os
import time
import pandas as pd
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from core.config import (
    STTM_DIR, LLM_PROVIDER,
    GROQ_API_KEY, GROQ_MODEL,
    GOOGLE_API_KEY, GEMINI_MODEL,
)
from core.audit import AuditLogger
from core.logger import log


# ---------------------------------------------------------------------------
# stdout/stderr safety wrapper
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
                log(f"[STTM] Rate limit hit, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    buf = io.StringIO()
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
# Pure Python context helpers (no LLM)
# ---------------------------------------------------------------------------

def _load_profile(profile_path: str) -> dict:
    with open(profile_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _summarise_bronze_context(profile_path: str) -> str:
    """Build a concise text summary of the data profile for the Bronze STTM prompt."""
    try:
        profile = _load_profile(profile_path)
    except Exception as exc:
        return f"Error loading profile: {exc}"
    lines = []
    for ds_name, ds_info in profile.get("datasets", {}).items():
        lines.append(f"Dataset: {ds_name}")
        shape = ds_info.get("shape", {})
        lines.append(f"  Rows: {shape.get('rows', '?')}, Columns: {shape.get('columns', '?')}")
        for col, info in ds_info.get("columns", {}).items():
            dtype = info.get("dtype", "?")
            null_pct = info.get("null_pct", 0)
            lines.append(f"  Column '{col}': dtype={dtype}, null_pct={null_pct}%")
    return "\n".join(lines)


def _summarise_parquet_context(parquet_paths: list) -> str:
    """Build a text summary of Parquet file schemas for Silver/Gold STTM prompts."""
    lines = []
    for fp in parquet_paths:
        try:
            df = pd.read_parquet(fp)
            name = os.path.splitext(os.path.basename(fp))[0]
            lines.append(f"Table: {name}")
            lines.append(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = int(df[col].isnull().sum())
                lines.append(f"  Column '{col}': dtype={dtype}, null_count={null_count}")
        except Exception as exc:
            lines.append(f"Error reading {fp}: {exc}")
    return "\n".join(lines)


def _parse_sttm_rows(text: str) -> list:
    """Extract a JSON array of STTM rows from raw LLM output."""
    for fence in ("```json", "```"):
        if fence in text:
            parts = text.split(fence)
            if len(parts) >= 3:
                text = parts[1]
                break
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        rows = json.loads(text[start:end + 1])
        if isinstance(rows, list):
            return rows
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Scratchpad shared between tools in the same run
# ---------------------------------------------------------------------------

_SCRATCHPAD: dict = {}


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_tools(context_paths: list, business_intent: str, layer: str, run_id: str):
    scratchpad: dict = {}

    @tool
    def inspect_context(confirmation: str = "execute") -> str:
        """Preview the source data context for the STTM layer being generated.

        For Bronze: reads the data profile JSON and returns column names, dtypes,
        null percentages, and statistics for every dataset.
        For Silver: reads Bronze Parquet files and returns schema metadata.
        For Gold: reads Silver Parquet files and returns schema metadata.
        Always call this FIRST before generating the STTM.
        Returns a JSON string with the context summary.
        """
        if layer == "bronze":
            # context_paths[0] is the profile JSON path for bronze
            if not context_paths:
                return json.dumps({"error": "No profile path provided for Bronze STTM"})
            try:
                profile = _load_profile(context_paths[0])
                return json.dumps({"layer": "bronze", "profile": profile}, default=str)
            except Exception as exc:
                return json.dumps({"error": str(exc)})
        else:
            # context_paths are Parquet files for silver/gold
            result = []
            for fp in context_paths:
                try:
                    df = pd.read_parquet(fp)
                    name = os.path.splitext(os.path.basename(fp))[0]
                    result.append({
                        "filename": os.path.basename(fp),
                        "table_name": name,
                        "columns": list(df.columns),
                        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                        "sample": df.head(3).to_dict(orient="records"),
                    })
                except Exception as exc:
                    result.append({"filename": os.path.basename(fp), "error": str(exc)})
            return json.dumps({"layer": layer, "tables": result}, default=str)

    @tool
    def generate_sttm(sttm_rows_json: str) -> str:
        """Save transformation rules as a STTM CSV and return the file path.

        Args:
            sttm_rows_json: A JSON array of STTM row objects. Each object must have:
                source_table, source_column, target_table, target_column,
                transformation_type, transformation_logic.

        transformation_type values allowed:
            rename, type_cast, fill_null, drop_null, dedup, date_format,
            join, aggregate, surrogate_key, passthrough

        Bronze rules: rename, type_cast, passthrough (intent-agnostic).
        Silver rules: fill_null, drop_null, dedup, date_format, type_cast (intent-agnostic).
        Gold rules: join, aggregate, rename, passthrough (intent-driven).

        Returns JSON: {"sttm_path": "...", "row_count": N}
        """
        rows = _parse_sttm_rows(sttm_rows_json)
        if not rows:
            # Attempt to parse the argument directly
            try:
                parsed = json.loads(sttm_rows_json)
                if isinstance(parsed, list):
                    rows = parsed
                elif isinstance(parsed, dict) and "rows" in parsed:
                    rows = parsed["rows"]
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_filename = f"sttm_{layer}_{run_id[:8]}.csv"
        sttm_path = str(STTM_DIR / sttm_filename)
        df = pd.DataFrame(rows)

        # Ensure required columns exist
        required_cols = [
            "source_table", "source_column", "target_table",
            "target_column", "transformation_type", "transformation_logic",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        df = df[required_cols]
        df.to_csv(sttm_path, index=False, encoding="utf-8")
        scratchpad["sttm_path"] = sttm_path
        log(f"[STTM] {layer.upper()} STTM saved -> {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    return [inspect_context, generate_sttm], scratchpad


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

BRONZE_SYSTEM_PROMPT = (
    "You are a data engineering specialist generating Bronze layer STTM rules. "
    "Bronze is intent-agnostic: map EVERY source column mechanically. "
    "\n\nRequired steps:"
    "\n1. Call inspect_context to see the data profile (columns, dtypes, stats)."
    "\n2. Generate a complete JSON array of STTM rows covering ALL source columns."
    "\n   - Use transformation_type: 'rename' for any column name standardisation."
    "\n   - Use transformation_type: 'type_cast' for type normalisation."
    "\n   - Use transformation_type: 'passthrough' for columns that need no change."
    "\n   - Add two metadata rows with target_column '_load_timestamp' (type_cast,"
    "     logic='Inject current UTC ISO timestamp at load time') and '_source_file'"
    "     (type_cast, logic='Inject source file path at load time')."
    "\n   - Do NOT add a surrogate key row -- that belongs in Silver."
    "\n3. Call generate_sttm with your JSON array as the sttm_rows_json argument."
    "\n\nRequired STTM row fields: source_table, source_column, target_table, "
    "target_column, transformation_type, transformation_logic."
    "\nUse only ASCII characters in all strings. No Unicode arrows or dashes."
)

SILVER_SYSTEM_PROMPT = (
    "You are a data engineering specialist generating Silver layer STTM rules. "
    "Silver is intent-agnostic: apply standard cleansing to EVERY Bronze column. "
    "\n\nRequired steps:"
    "\n1. Call inspect_context to see the Bronze Parquet schemas."
    "\n2. Generate a complete JSON array of STTM rows:"
    "\n   - FIRST row must be the surrogate key: source_column='', "
    "     target_column='pk_<table_stem>_silver_id', "
    "     transformation_type='surrogate_key', "
    "     transformation_logic='Auto-generated sequential surrogate primary key starting from 1'."
    "\n   - For each column apply appropriate cleansing:"
    "\n     * Numeric columns: fill_null with mean or median as appropriate."
    "\n     * Text/category columns: fill_null with mode or empty string."
    "\n     * Date columns: date_format to YYYY-MM-DD."
    "\n     * ID columns: type_cast only, no null handling."
    "\n     * All tables: add one dedup row with transformation_logic='Drop fully duplicate rows'."
    "\n3. Call generate_sttm with your JSON array."
    "\n\nRequired STTM row fields: source_table, source_column, target_table, "
    "target_column, transformation_type, transformation_logic."
    "\nUse only ASCII characters. No Unicode arrows or dashes."
)

GOLD_SYSTEM_PROMPT = (
    "You are a data engineering specialist generating Gold layer STTM rules. "
    "Gold is intent-driven: shape the output tables to answer the business question. "
    "\n\nRequired steps:"
    "\n1. Call inspect_context to see the Silver Parquet schemas."
    "\n2. Generate a complete JSON array of STTM rows:"
    "\n   - FIRST row must be the surrogate key: source_column='', "
    "     target_column='pk_gold_id', transformation_type='surrogate_key', "
    "     transformation_logic='Auto-generated sequential surrogate primary key starting from 1'."
    "\n   - Include join rules where Silver tables share key columns (_id suffix)."
    "\n   - Include aggregate rules (sum/avg/count/max/min) for metrics relevant to the intent."
    "\n   - Include passthrough/rename rules for dimension columns needed by the Reporter."
    "\n   - Preserve numeric columns needed for aggregation -- do NOT drop them."
    "\n3. Call generate_sttm with your JSON array."
    "\n\nRequired STTM row fields: source_table, source_column, target_table, "
    "target_column, transformation_type, transformation_logic."
    "\nUse only ASCII characters. No Unicode arrows or dashes."
)

_SYSTEM_PROMPTS = {
    "bronze": BRONZE_SYSTEM_PROMPT,
    "silver": SILVER_SYSTEM_PROMPT,
    "gold": GOLD_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sttm(context_paths: list, business_intent: str, layer: str, run_id: str) -> str:
    """Generate a STTM CSV for the specified Medallion layer and return its path.

    Args:
        context_paths: For 'bronze': list with one profile JSON path.
                       For 'silver': list of Bronze Parquet paths.
                       For 'gold': list of Silver Parquet paths.
        business_intent: The user's analytical question (used only for Gold).
        layer: One of 'bronze', 'silver', 'gold'.
        run_id: Unique identifier for this pipeline run.

    Returns:
        str: Absolute path to the saved STTM CSV file.
    """
    if layer not in ("bronze", "silver", "gold"):
        raise ValueError(f"Invalid layer '{layer}'. Must be bronze, silver, or gold.")

    audit = AuditLogger(run_id)
    audit.log("started", "sttm", {"layer": layer, "context_paths": context_paths})
    log(f"[STTM] Starting {layer.upper()} STTM for run_id={run_id}")

    tools, scratchpad = _make_tools(context_paths, business_intent, layer, run_id)
    system_prompt = _SYSTEM_PROMPTS[layer]

    human_message = (
        f"Generate a complete {layer.upper()} STTM for run_id={run_id}.\n"
        f"Context paths: {context_paths}\n"
    )
    if layer == "gold":
        human_message += f"Business intent: {business_intent}\n"
    human_message += (
        "Step 1: Call inspect_context to understand the available columns.\n"
        f"Step 2: Call generate_sttm with a complete JSON array of {layer} STTM rows."
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        llm = _make_llm()
        agent = create_react_agent(llm, tools, prompt=system_prompt)

    try:
        result = _invoke_agent(agent, {"messages": [HumanMessage(content=human_message)]})
    except Exception as exc:
        audit.log("failed", "sttm", {"layer": layer, "error": str(exc)})
        log(f"[STTM] Agent failed: {exc}. Generating fallback STTM.")
        return _fallback_sttm(context_paths, layer, run_id, business_intent)

    # Extract sttm_path from scratchpad (set by generate_sttm tool)
    sttm_path = scratchpad.get("sttm_path", "")

    # Fallback: scan messages for the path
    if not sttm_path:
        messages = result.get("messages", [])
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
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
                if isinstance(parsed, dict) and "sttm_path" in parsed:
                    sttm_path = parsed["sttm_path"]
                    break
            except (json.JSONDecodeError, ValueError):
                pass
            if f"sttm_{layer}" in content:
                for word in content.split():
                    word = word.strip('",\'')
                    if f"sttm_{layer}" in word and word.endswith(".csv"):
                        sttm_path = word
                        break
            if sttm_path:
                break

    # If still no valid path, run fallback
    if not sttm_path or not Path(sttm_path).exists():
        log(f"[STTM] No valid STTM path found, using fallback for {layer}")
        sttm_path = _fallback_sttm(context_paths, layer, run_id, business_intent)

    audit.log("completed", "sttm", {"layer": layer, "sttm_path": sttm_path})
    log(f"[STTM] {layer.upper()} STTM done -> {sttm_path}")
    return sttm_path


def _fallback_sttm(context_paths: list, layer: str, run_id: str, business_intent: str = "") -> str:
    """Generate a minimal but valid STTM CSV without LLM when the agent fails."""
    rows = []
    sttm_filename = f"sttm_{layer}_{run_id[:8]}.csv"
    sttm_path = str(STTM_DIR / sttm_filename)

    if layer == "bronze":
        if context_paths:
            try:
                with open(context_paths[0], "r", encoding="utf-8") as fh:
                    profile = json.load(fh)
                for ds_name, ds_info in profile.get("datasets", {}).items():
                    for col in ds_info.get("columns", {}).keys():
                        rows.append({
                            "source_table": ds_name,
                            "source_column": col,
                            "target_table": ds_name + "_bronze",
                            "target_column": col,
                            "transformation_type": "passthrough",
                            "transformation_logic": "Direct copy of source column",
                        })
                    rows.append({
                        "source_table": ds_name, "source_column": "",
                        "target_table": ds_name + "_bronze", "target_column": "_load_timestamp",
                        "transformation_type": "type_cast",
                        "transformation_logic": "Inject current UTC ISO timestamp at load time",
                    })
                    rows.append({
                        "source_table": ds_name, "source_column": "",
                        "target_table": ds_name + "_bronze", "target_column": "_source_file",
                        "transformation_type": "type_cast",
                        "transformation_logic": "Inject source file path at load time",
                    })
            except Exception as exc:
                log(f"[STTM] Fallback bronze failed: {exc}")

    elif layer == "silver":
        for fp in context_paths:
            try:
                df = pd.read_parquet(fp)
                stem = os.path.splitext(os.path.basename(fp))[0]
                table_stem = stem.replace("_bronze", "")
                rows.append({
                    "source_table": stem, "source_column": "",
                    "target_table": table_stem + "_silver",
                    "target_column": f"pk_{table_stem}_silver_id",
                    "transformation_type": "surrogate_key",
                    "transformation_logic": "Auto-generated sequential surrogate primary key starting from 1",
                })
                for col in df.columns:
                    rows.append({
                        "source_table": stem, "source_column": col,
                        "target_table": table_stem + "_silver", "target_column": col,
                        "transformation_type": "passthrough",
                        "transformation_logic": "Direct copy from Bronze",
                    })
                rows.append({
                    "source_table": stem, "source_column": "",
                    "target_table": table_stem + "_silver", "target_column": "",
                    "transformation_type": "dedup",
                    "transformation_logic": "Drop fully duplicate rows",
                })
            except Exception as exc:
                log(f"[STTM] Fallback silver failed for {fp}: {exc}")

    elif layer == "gold":
        rows.append({
            "source_table": "", "source_column": "",
            "target_table": "gold_table", "target_column": "pk_gold_id",
            "transformation_type": "surrogate_key",
            "transformation_logic": "Auto-generated sequential surrogate primary key starting from 1",
        })
        for fp in context_paths:
            try:
                df = pd.read_parquet(fp)
                stem = os.path.splitext(os.path.basename(fp))[0]
                for col in df.columns:
                    rows.append({
                        "source_table": stem, "source_column": col,
                        "target_table": "gold_table", "target_column": col,
                        "transformation_type": "passthrough",
                        "transformation_logic": "Direct copy from Silver",
                    })
            except Exception as exc:
                log(f"[STTM] Fallback gold failed for {fp}: {exc}")

    pd.DataFrame(rows).to_csv(sttm_path, index=False, encoding="utf-8")
    log(f"[STTM] Fallback {layer.upper()} STTM saved -> {sttm_path} ({len(rows)} rows)")
    return sttm_path
