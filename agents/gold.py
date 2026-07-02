"""Gold layer execution agent -- pure Python, no LLM.

Reads Silver Parquet files, applies approved STTM materialisation rules,
joins tables on matching _id columns, applies aggregations, and writes
Parquet files to the Gold layer (one file per unique target_table in STTM).

Public API:
    run_gold(input_files, sttm_path, run_id) -> list[str]
"""

import os
import pandas as pd
from pathlib import Path

from core.config import GOLD_DIR
from core.audit import AuditLogger
from core.logger import log


def run_gold(input_files: list, sttm_path: str, run_id: str) -> list:
    """Materialise Gold tables from Silver Parquet files using STTM rules.

    Supported transformation_type values:
        - join: outer join tables on matching _id columns
        - aggregate: group-by + aggregation (sum/avg/count/max/min)
        - rename: rename source_column to target_column
        - passthrough: include column as-is
        - surrogate_key: inject pk_gold_id as first column

    One output Parquet file is produced per unique target_table in the STTM.

    Args:
        input_files: List of Silver Parquet file paths.
        sttm_path: Path to the approved Gold STTM CSV.
        run_id: Unique identifier for this pipeline run.

    Returns:
        list: Absolute paths to the written Gold Parquet files.
    """
    audit = AuditLogger(run_id)
    audit.log("started", "gold", {"input_files": input_files, "sttm_path": sttm_path})
    log(f"[GOLD] Starting run_id={run_id}, files={input_files}")

    sttm_df = pd.read_csv(sttm_path).fillna("")
    sttm_df.columns = [c.strip() for c in sttm_df.columns]

    # Load all Silver files keyed by table stem
    source_tables: dict = {}
    for fp in input_files:
        try:
            df = pd.read_parquet(fp)
            stem = os.path.splitext(os.path.basename(fp))[0]
            # Normalise stem: strip _silver suffix for matching
            norm_stem = stem.replace("_silver", "")
            source_tables[stem] = df
            source_tables[norm_stem] = df
            log(f"[GOLD] Loaded Silver table '{stem}': {df.shape}")
        except Exception as exc:
            log(f"[GOLD] Failed to load {fp}: {exc}")

    if not source_tables:
        log("[GOLD] No Silver tables loaded -- aborting")
        audit.log("failed", "gold", {"reason": "No Silver tables could be loaded"})
        return []

    # Group STTM rules by target_table
    if "target_table" not in sttm_df.columns:
        sttm_df["target_table"] = "gold_table"

    output_paths = []

    for target_table_name, table_rules in sttm_df.groupby("target_table"):
        target_table_name = str(target_table_name).strip()
        if not target_table_name:
            continue

        log(f"[GOLD] Building target table '{target_table_name}' ({len(table_rules)} rules)")

        # Collect unique non-empty source tables referenced by rules
        source_names_needed = [
            str(s).strip()
            for s in table_rules["source_table"].unique()
            if str(s).strip()
        ]

        # Resolve Silver DataFrames for each source name
        available: dict = {}
        for sn in source_names_needed:
            if sn in source_tables:
                available[sn] = source_tables[sn]
            else:
                # Try stripping common suffixes for matching
                for variant in (sn.replace("_silver", ""), sn + "_silver"):
                    if variant in source_tables:
                        available[sn] = source_tables[variant]
                        break

        if not available:
            log(f"[GOLD] No source data found for '{target_table_name}', using all Silver tables")
            # Use first available Silver table as fallback
            if source_tables:
                first_key = next(iter(source_tables))
                available[first_key] = source_tables[first_key]

        # Start with the first available table
        first_key = next(iter(available))
        df = available[first_key].copy()

        # Join additional tables (outer join on matching _id columns)
        metadata_cols = {"_load_timestamp", "_source_file"}
        for src_name, src_df in list(available.items())[1:]:
            common = [
                c for c in df.columns
                if c in src_df.columns
                and (c.endswith("_id") or c == "id")
                and c not in metadata_cols
            ]
            if common:
                df = df.merge(src_df, on=common, how="outer", suffixes=("", "_dup"))
                dup_cols = [c for c in df.columns if c.endswith("_dup")]
                if dup_cols:
                    df = df.drop(columns=dup_cols)
            else:
                df = pd.concat([df, src_df], ignore_index=True, sort=False)

        # Apply transformations from rules
        rename_map: dict = {}
        group_by_cols: list = []
        agg_map: dict = {}

        for _, rule in table_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            t_type = str(rule.get("transformation_type", "")).strip().lower()
            logic = str(rule.get("transformation_logic", "")).lower()

            if t_type == "surrogate_key":
                continue

            # Collect renames
            if source_col and target_col and source_col != target_col and source_col in df.columns:
                rename_map[source_col] = target_col

            # Collect group-by columns (passthrough/rename = dimension)
            working = source_col
            if t_type in ("passthrough", "rename", "direct"):
                if working and working in df.columns and working not in group_by_cols:
                    group_by_cols.append(working)

            # Collect aggregation rules
            if t_type == "aggregate" and source_col and source_col in df.columns:
                if "sum" in logic:
                    agg_map[source_col] = "sum"
                elif any(kw in logic for kw in ("avg", "average", "mean")):
                    agg_map[source_col] = "mean"
                elif "count" in logic:
                    agg_map[source_col] = "count"
                elif "max" in logic:
                    agg_map[source_col] = "max"
                elif "min" in logic:
                    agg_map[source_col] = "min"

        # Apply renames
        valid_renames = {s: t for s, t in rename_map.items() if s in df.columns}
        if valid_renames:
            df = df.rename(columns=valid_renames)
            # Update group_by and agg_map after rename
            group_by_cols = [valid_renames.get(c, c) for c in group_by_cols]
            agg_map = {valid_renames.get(c, c): fn for c, fn in agg_map.items()}

        # Apply aggregation
        valid_group_by = [c for c in group_by_cols if c in df.columns]
        valid_agg = {c: fn for c, fn in agg_map.items() if c in df.columns and c not in valid_group_by}
        if valid_group_by and valid_agg:
            df = df.groupby(valid_group_by, dropna=False, as_index=False).agg(valid_agg)

        # Filter to approved target columns + system columns
        approved_targets = set(
            table_rules["target_column"].dropna().astype(str).str.strip().tolist()
        )
        columns_to_keep = [
            c for c in df.columns
            if c in approved_targets or c.startswith("_") or c.startswith("pk_")
        ]
        if columns_to_keep:
            df = df[columns_to_keep]

        # Inject surrogate key as first column
        pk_col = "pk_gold_id"
        if pk_col not in df.columns:
            df.insert(0, pk_col, range(1, len(df) + 1))
        else:
            cols = [pk_col] + [c for c in df.columns if c != pk_col]
            df = df[cols]

        # Write Parquet
        safe_name = target_table_name.replace(" ", "_")
        output_path = str(GOLD_DIR / f"{safe_name}.parquet")
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        log(f"[GOLD] Created '{target_table_name}': {df.shape[0]} rows x {df.shape[1]} cols")
        audit.log("table_created", "gold", {
            "target_table": target_table_name,
            "output_file": output_path,
            "shape": list(df.shape),
        })

    # Fallback: if STTM grouping produced nothing, write a simple pass-through Gold table
    if not output_paths and source_tables:
        log("[GOLD] No STTM rules produced output -- writing fallback Gold table")
        frames = list(source_tables.values())
        combined = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0].copy()
        combined.insert(0, "pk_gold_id", range(1, len(combined) + 1))
        output_path = str(GOLD_DIR / "gold_table.parquet")
        combined.to_parquet(output_path, index=False)
        output_paths.append(output_path)
        audit.log("fallback_created", "gold", {"output_file": output_path})

    audit.log("completed", "gold", {"output_files": output_paths})
    log(f"[GOLD] Done -> {output_paths}")
    return output_paths
