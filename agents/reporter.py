"""Reporter agent - pure Python, no LLM.

Reads Gold Parquet files, runs automatic DuckDB analysis,
and generates a self-contained HTML report with Plotly charts.

Public API:
    run_reporter(gold_files, business_intent, run_id) -> str  (HTML report path)
"""

import json
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

from core.config import REPORTS_DIR
from core.audit import AuditLogger
from core.logger import log


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def _load_tables(gold_files: list) -> tuple:
    """Load Gold Parquet files into DuckDB. Returns (conn, table_map)."""
    conn = duckdb.connect()
    table_map = {}
    for fp in gold_files:
        stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
        conn.execute(f"CREATE TABLE {stem} AS SELECT * FROM read_parquet('{fp}')")
        table_map[stem] = fp
    return conn, table_map


def _auto_analyze(conn: duckdb.DuckDBPyConnection, table_map: dict, business_intent: str) -> dict:
    """Run automatic analysis on Gold tables. Returns analysis dict."""
    results = {}
    for tname in table_map:
        info = {}
        schema = conn.execute(f"DESCRIBE {tname}").fetchdf()
        info["schema"] = schema.to_dict(orient="records")
        info["row_count"] = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]

        numeric_cols = [r["column_name"] for r in info["schema"]
                        if any(t in r["column_type"].upper()
                               for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "BIGINT", "HUGEINT"))]
        text_cols = [r["column_name"] for r in info["schema"]
                     if any(t in r["column_type"].upper() for t in ("VARCHAR", "TEXT", "CHAR"))]

        # Summary stats for numeric columns
        if numeric_cols:
            parts = []
            for i, c in enumerate(numeric_cols[:5]):
                parts.append(f'SUM("{c}") AS sum_{i}')
                parts.append(f'AVG("{c}") AS avg_{i}')
                parts.append(f'MIN("{c}") AS min_{i}')
                parts.append(f'MAX("{c}") AS max_{i}')

            select_clause = ", ".join(parts)
            stats_sql = f'SELECT {select_clause} FROM "{tname}"'
            stats = conn.execute(stats_sql).fetchdf()
            info["numeric_summary"] = stats.to_dict(orient="records")

        # Top groupings for first text column
        if text_cols and numeric_cols:
            grp = conn.execute(
                f'SELECT "{text_cols[0]}", SUM("{numeric_cols[0]}") as total '
                f'FROM {tname} GROUP BY "{text_cols[0]}" ORDER BY total DESC LIMIT 15'
            ).fetchdf()
            info["top_groups"] = grp.to_dict(orient="records")
            info["group_col"] = text_cols[0]
            info["value_col"] = numeric_cols[0]

        results[tname] = info
    return results


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _make_charts(conn, table_map: dict, analysis: dict) -> list:
    """Build a list of Plotly HTML chart strings."""
    charts = []

    for tname, info in analysis.items():
        df = conn.execute(f"SELECT * FROM {tname}").fetchdf()
        if df.empty:
            continue

        schema = info["schema"]
        numeric_cols = [r["column_name"] for r in schema
                        if any(t in r["column_type"].upper()
                               for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "BIGINT", "HUGEINT"))]
        text_cols = [r["column_name"] for r in schema
                     if any(t in r["column_type"].upper() for t in ("VARCHAR", "TEXT", "CHAR"))]

        # Bar chart: top groups
        if text_cols and numeric_cols:
            gcol, vcol = text_cols[0], numeric_cols[0]
            grp = df.groupby(gcol, dropna=False)[vcol].sum().sort_values(ascending=False).head(15)
            fig = go.Figure(go.Bar(
                x=grp.index.astype(str).tolist(),
                y=grp.values.tolist(),
                marker_color="#667eea",
            ))
            fig.update_layout(
                title=f"{vcol} by {gcol}",
                xaxis_title=gcol, yaxis_title=vcol,
                height=420, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=80),
            )
            charts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

        # Second numeric column line/bar
        if len(numeric_cols) >= 2 and text_cols:
            gcol, vcol = text_cols[0], numeric_cols[1]
            grp2 = df.groupby(gcol, dropna=False)[vcol].sum().sort_values(ascending=False).head(15)
            fig2 = go.Figure(go.Bar(
                x=grp2.index.astype(str).tolist(),
                y=grp2.values.tolist(),
                marker_color="#f093fb",
            ))
            fig2.update_layout(
                title=f"{vcol} by {gcol}",
                xaxis_title=gcol, yaxis_title=vcol,
                height=420, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=80),
            )
            charts.append(fig2.to_html(include_plotlyjs=False, full_html=False))

        # Pie chart if text + numeric
        if text_cols and numeric_cols:
            gcol, vcol = text_cols[0], numeric_cols[0]
            pie_df = df.groupby(gcol, dropna=False)[vcol].sum()
            fig3 = go.Figure(go.Pie(
                labels=pie_df.index.astype(str).tolist(),
                values=pie_df.values.tolist(),
                hole=0.35,
            ))
            fig3.update_layout(
                title=f"{vcol} distribution by {gcol}",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            charts.append(fig3.to_html(include_plotlyjs=False, full_html=False))

        # Numeric distribution histogram
        if numeric_cols:
            vcol = numeric_cols[0]
            fig4 = px.histogram(df, x=vcol, title=f"Distribution of {vcol}", template="plotly_white")
            fig4.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
            charts.append(fig4.to_html(include_plotlyjs=False, full_html=False))

    return charts


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(analysis: dict, business_intent: str) -> dict:
    """Build a plain-text summary from analysis results."""
    lines = []
    for tname, info in analysis.items():
        lines.append(f"Table: {tname} | Rows: {info['row_count']}")
        if "top_groups" in info and info["top_groups"]:
            gcol = info["group_col"]
            vcol = info["value_col"]
            top = info["top_groups"][0]
            lines.append(f"  Top {gcol}: {top.get(gcol, 'N/A')} with {vcol}={top.get('total', 'N/A'):.2f}")
        if "numeric_summary" in info:
            lines.append(f"  Numeric columns analyzed: {len(info.get('schema', []))} columns")
    return {
        "business_intent": business_intent,
        "tables_analyzed": list(analysis.keys()),
        "summary_lines": lines,
    }


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def _build_html(run_id: str, business_intent: str, summary: dict, charts: list) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    intent_safe = business_intent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    summary_html = "".join(f"<li>{line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</li>"
                           for line in summary.get("summary_lines", []))
    tables_html = ", ".join(summary.get("tables_analyzed", []))
    charts_html = "\n".join(f'<div class="chart-card">{c}</div>' for c in charts) if charts else "<p>No charts generated.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IDAMP Report - {run_id[:8]}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#1a1a2e;padding:24px}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:28px 32px;border-radius:14px;margin-bottom:24px}}
.header h1{{font-size:1.8rem;margin-bottom:6px}}
.header p{{opacity:.85;font-size:.95rem}}
.meta{{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}}
.badge{{background:rgba(255,255,255,.2);padding:4px 12px;border-radius:20px;font-size:.8rem}}
.section{{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{font-size:1.1rem;color:#667eea;margin-bottom:14px;border-bottom:2px solid #f0f2f5;padding-bottom:8px}}
.summary-list{{list-style:none;line-height:2}}
.summary-list li{{padding:4px 0;border-bottom:1px solid #f5f5f5;font-size:.92rem}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}}
.chart-card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.footer{{text-align:center;color:#888;font-size:.8rem;margin-top:24px}}
</style>
</head>
<body>
<div class="header">
  <h1>Executive Report</h1>
  <p><strong>Business Intent:</strong> {intent_safe}</p>
  <div class="meta">
    <span class="badge">Run: {run_id[:8]}</span>
    <span class="badge">Generated: {ts}</span>
    <span class="badge">Tables: {tables_html}</span>
  </div>
</div>

<div class="section">
  <h2>Analysis Summary</h2>
  <ul class="summary-list">{summary_html}</ul>
</div>

<div class="section">
  <h2>Charts</h2>
  <div class="chart-grid">
    {charts_html}
  </div>
</div>

<div class="footer">
  Generated by IDAMP - Intent-Driven Agentic Medallion Pipeline
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_reporter(gold_files: list, business_intent: str, run_id: str) -> str:
    """Generate an HTML report from Gold Parquet files.

    Pure Python implementation - no LLM required.
    Reads Gold data, runs automatic DuckDB analysis, builds Plotly charts.

    Returns:
        str: Path to the saved HTML report file.
    """
    audit = AuditLogger(run_id)
    audit.log("started", "reporter", {"gold_files": gold_files, "intent": business_intent})
    log(f"[REPORTER] Starting run_id={run_id} gold_files={gold_files}")

    try:
        conn, table_map = _load_tables(gold_files)
        log(f"[REPORTER] Loaded tables: {list(table_map.keys())}")

        analysis = _auto_analyze(conn, table_map, business_intent)
        log("[REPORTER] Analysis complete")

        charts = _make_charts(conn, table_map, analysis)
        log(f"[REPORTER] Built {len(charts)} charts")

        summary = _build_summary(analysis, business_intent)
        html_content = _build_html(run_id, business_intent, summary, charts)

        report_path = str(REPORTS_DIR / f"report_{run_id[:8]}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        json_path = str(REPORTS_DIR / f"report_{run_id[:8]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "analysis": {k: {kk: vv for kk, vv in v.items() if kk != "numeric_summary"} for k, v in analysis.items()}}, f, indent=2, default=str)

        conn.close()
        audit.log("completed", "reporter", {"report_path": report_path})
        log(f"[REPORTER] Report saved -> {report_path}")
        return report_path

    except Exception as exc:
        import traceback
        err = f"Reporter failed: {exc}\n{traceback.format_exc()}"
        audit.log("failed", "reporter", {"error": str(exc)})
        log(f"[REPORTER] FAILED: {exc}")
        raise RuntimeError(err) from exc
