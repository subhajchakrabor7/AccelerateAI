"""Bronze layer execution agent -- pure Python, no LLM.

Reads raw CSV files, applies approved STTM rules (rename, type_cast, passthrough),
injects lineage metadata (_load_timestamp, _source_file), and writes Parquet files
to the Bronze layer.

Public API:
    run_bronze(input_files, sttm_path, run_id) -> list[str]
"""

import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from core.config import BRONZE_DIR
from core.audit import AuditLogger
from core.logger import log


def run_bronze(input_files: list, sttm_path: str, run_id: str) -> list:
    """Apply Bronze STTM rules to raw CSV files and write Parquet output.

    Supported transformation_type values:
        - rename: rename source_column to target_column
        - type_cast: cast column to type specified in transformation_logic
        - passthrough: keep column as-is

    Metadata columns always added:
        - _load_timestamp: UTC ISO timestamp string
        - _source_file: source file name

    Args:
        input_files: List of raw CSV file paths.
        sttm_path: Path to the approved Bronze STTM CSV.
        run_id: Unique identifier for this pipeline run.

    Returns:
        list: Absolute paths to the written Bronze Parquet files.
    """
    audit = AuditLogger(run_id)
    audit.log("started", "bronze", {"input_files": input_files, "sttm_path": sttm_path})
    log(f"[BRONZE] Starting run_id={run_id}, files={input_files}")

    sttm_df = pd.read_csv(sttm_path).fillna("")
    output_paths = []

    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            log(f"[BRONZE] Failed to read {file_path}: {exc}")
            audit.log("file_error", "bronze", {"file": file_path, "error": str(exc)})
            continue

        original_shape = df.shape
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]

        # Filter STTM rules to those matching this file (or global rules with empty source_table)
        if "source_table" in sttm_df.columns:
            file_rules = sttm_df[
                sttm_df["source_table"].astype(str).str.strip().isin(["", file_name, file_stem])
            ]
        else:
            file_rules = sttm_df

        for _, rule in file_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            t_type = str(rule.get("transformation_type", "")).strip().lower()
            logic = str(rule.get("transformation_logic", "")).lower()

            # Handle metadata injections regardless of transformation_type
            if target_col.lower() in ("_load_timestamp", "load_timestamp"):
                df[target_col] = datetime.now(timezone.utc).isoformat()
                continue
            if target_col.lower() in ("_source_file", "source_file"):
                df[target_col] = file_name
                continue

            # Rename rule: rename source_col to target_col
            if t_type == "rename" and source_col and target_col:
                if source_col in df.columns and source_col != target_col:
                    df = df.rename(columns={source_col: target_col})
                continue

            # Passthrough: no transformation needed
            if t_type == "passthrough":
                continue

            # Type cast: determine working column after any rename
            working_col = target_col if target_col in df.columns else source_col
            if not working_col or working_col not in df.columns:
                continue

            if t_type == "type_cast" or t_type == "rename":
                try:
                    if any(kw in logic for kw in ("str", "text", "string", "varchar")):
                        df[working_col] = df[working_col].astype(str)
                    elif any(kw in logic for kw in ("int", "integer", "whole")):
                        df[working_col] = pd.to_numeric(df[working_col], errors="coerce").astype("Int64")
                    elif any(kw in logic for kw in ("float", "decimal", "numeric", "double")):
                        df[working_col] = pd.to_numeric(df[working_col], errors="coerce")
                    elif any(kw in logic for kw in ("date", "datetime", "timestamp")):
                        df[working_col] = pd.to_datetime(df[working_col], errors="coerce")
                except (ValueError, TypeError) as exc:
                    log(f"[BRONZE] Type cast failed for {working_col}: {exc}")

        # Always inject metadata columns if not already present
        if "_load_timestamp" not in df.columns:
            df["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
        if "_source_file" not in df.columns:
            df["_source_file"] = file_name

        # Write Parquet
        output_filename = file_stem + "_bronze.parquet"
        output_path = str(BRONZE_DIR / output_filename)
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        log(f"[BRONZE] {file_name} -> {output_filename} ({df.shape[0]} rows x {df.shape[1]} cols)")
        audit.log("file_processed", "bronze", {
            "input_file": file_path,
            "output_file": output_path,
            "input_shape": list(original_shape),
            "output_shape": list(df.shape),
        })

    audit.log("completed", "bronze", {"output_files": output_paths})
    log(f"[BRONZE] Done -> {output_paths}")
    return output_paths
