"""Silver layer execution agent -- pure Python, no LLM.

Reads Bronze Parquet files, applies approved STTM cleansing rules, injects
surrogate keys, and writes Parquet files to the Silver layer.

Public API:
    run_silver(input_files, sttm_path, run_id) -> list[str]
"""

import os
import pandas as pd
from pathlib import Path

from core.config import SILVER_DIR
from core.audit import AuditLogger
from core.logger import log


# ---------------------------------------------------------------------------
# Date format candidates for standardisation
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y%m%d", "%d-%b-%Y", "%d-%B-%Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%m-%d-%Y", "%d.%m.%Y",
]


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Try multiple date formats; return a datetime Series or coerce to NaT."""
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(series, errors="coerce")


def run_silver(input_files: list, sttm_path: str, run_id: str) -> list:
    """Apply Silver STTM cleansing rules to Bronze Parquet files and write Silver Parquet.

    Supported transformation_type values:
        - fill_null: fill nulls using transformation_logic (mean/median/mode/constant)
        - drop_null: drop rows where column is null
        - dedup: drop duplicate rows
        - date_format: standardise date column to YYYY-MM-DD string
        - type_cast: cast column type (int/float/str)
        - surrogate_key: inject pk_<stem>_silver_id as first column
        - passthrough: keep as-is (with optional rename)
        - rename: rename source_column to target_column

    Args:
        input_files: List of Bronze Parquet file paths.
        sttm_path: Path to the approved Silver STTM CSV.
        run_id: Unique identifier for this pipeline run.

    Returns:
        list: Absolute paths to the written Silver Parquet files.
    """
    audit = AuditLogger(run_id)
    audit.log("started", "silver", {"input_files": input_files, "sttm_path": sttm_path})
    log(f"[SILVER] Starting run_id={run_id}, files={input_files}")

    sttm_df = pd.read_csv(sttm_path).fillna("")
    # Normalise column names
    sttm_df.columns = [c.strip() for c in sttm_df.columns]
    output_paths = []

    for file_path in input_files:
        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            log(f"[SILVER] Failed to read {file_path}: {exc}")
            audit.log("file_error", "silver", {"file": file_path, "error": str(exc)})
            continue

        original_shape = df.shape
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]
        # Derive cleaner stem for surrogate key naming
        table_stem = file_stem.replace("_bronze", "")

        # Filter rules to this file or global (empty source_table)
        if "source_table" in sttm_df.columns:
            file_rules = sttm_df[
                sttm_df["source_table"].astype(str).str.strip().isin(["", file_name, file_stem, table_stem])
            ]
        else:
            file_rules = sttm_df

        # Detect surrogate key target name from STTM (first surrogate_key row)
        pk_col = f"pk_{table_stem}_silver_id"
        surr_rows = file_rules[
            file_rules.get("transformation_type", pd.Series(dtype=str)).astype(str).str.strip().str.lower() == "surrogate_key"
        ]
        if not surr_rows.empty:
            declared_pk = str(surr_rows.iloc[0].get("target_column", "")).strip()
            if declared_pk:
                pk_col = declared_pk

        # Apply rules
        for _, rule in file_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            t_type = str(rule.get("transformation_type", "")).strip().lower()
            logic = str(rule.get("transformation_logic", "")).lower()

            # Skip surrogate_key rows -- handled below
            if t_type == "surrogate_key":
                continue

            # Rename first
            if source_col and target_col and source_col in df.columns and source_col != target_col:
                df = df.rename(columns={source_col: target_col})

            # Determine working column
            working_col = target_col if target_col in df.columns else source_col

            # dedup -- applied at DataFrame level, not per-column
            if t_type == "dedup":
                df = df.drop_duplicates()
                continue

            if not working_col or working_col not in df.columns:
                continue

            try:
                if t_type == "drop_null":
                    df = df.dropna(subset=[working_col])

                elif t_type == "fill_null":
                    if "mean" in logic:
                        fill_val = pd.to_numeric(df[working_col], errors="coerce").mean()
                        df[working_col] = df[working_col].fillna(fill_val)
                    elif "median" in logic:
                        fill_val = pd.to_numeric(df[working_col], errors="coerce").median()
                        df[working_col] = df[working_col].fillna(fill_val)
                    elif "mode" in logic:
                        mode_s = df[working_col].mode()
                        if not mode_s.empty:
                            df[working_col] = df[working_col].fillna(mode_s.iloc[0])
                    else:
                        # Fill with empty string or 0 depending on dtype
                        if pd.api.types.is_numeric_dtype(df[working_col]):
                            df[working_col] = df[working_col].fillna(0)
                        else:
                            df[working_col] = df[working_col].fillna("")

                elif t_type == "date_format":
                    parsed = _parse_date_column(df[working_col])
                    df[working_col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other=None)

                elif t_type == "type_cast":
                    if any(kw in logic for kw in ("int", "integer", "whole")):
                        df[working_col] = pd.to_numeric(df[working_col], errors="coerce").astype("Int64")
                    elif any(kw in logic for kw in ("float", "decimal", "numeric", "double")):
                        df[working_col] = pd.to_numeric(df[working_col], errors="coerce")
                    elif any(kw in logic for kw in ("str", "text", "string", "varchar")):
                        df[working_col] = df[working_col].astype(str)
                    elif any(kw in logic for kw in ("date", "datetime")):
                        parsed = _parse_date_column(df[working_col])
                        df[working_col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other=None)

                # Text normalisation within any rule
                if "lowercase" in logic:
                    df[working_col] = df[working_col].astype(str).str.lower()
                elif "uppercase" in logic:
                    df[working_col] = df[working_col].astype(str).str.upper()
                if "strip" in logic or "trim" in logic:
                    if pd.api.types.is_string_dtype(df[working_col]):
                        df[working_col] = df[working_col].str.strip()

            except (ValueError, TypeError, AttributeError) as exc:
                log(f"[SILVER] Rule application error for {working_col}: {exc}")

        # Inject surrogate primary key as first column
        if pk_col not in df.columns:
            df.insert(0, pk_col, range(1, len(df) + 1))
        else:
            # Move to front
            cols = [pk_col] + [c for c in df.columns if c != pk_col]
            df = df[cols]

        # Filter to approved target columns + system columns
        if "target_column" in file_rules.columns:
            approved_cols = set(file_rules["target_column"].dropna().astype(str).str.strip().unique())
        else:
            approved_cols = set()

        columns_to_keep = [
            c for c in df.columns
            if c in approved_cols or c.startswith("_") or c.startswith("pk_")
        ]
        # Ensure pk_col is always first
        if pk_col in columns_to_keep:
            columns_to_keep = [pk_col] + [c for c in columns_to_keep if c != pk_col]
        # Fall back to all columns if filter would leave nothing
        if not columns_to_keep:
            columns_to_keep = list(df.columns)
        df = df[columns_to_keep]

        # Write Parquet
        output_filename = file_stem.replace("_bronze", "") + "_silver.parquet"
        output_path = str(SILVER_DIR / output_filename)
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        log(f"[SILVER] {file_name} -> {output_filename} ({df.shape[0]} rows x {df.shape[1]} cols)")
        audit.log("file_processed", "silver", {
            "input_file": file_path,
            "output_file": output_path,
            "input_shape": list(original_shape),
            "output_shape": list(df.shape),
        })

    audit.log("completed", "silver", {"output_files": output_paths})
    log(f"[SILVER] Done -> {output_paths}")
    return output_paths
