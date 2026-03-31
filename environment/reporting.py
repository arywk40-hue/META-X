"""Dataset profiling, work-queue generation, and report artifacts."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


MAX_TOP_COLUMNS = 12
BATCH_SIZE = 10


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "dataset"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _value_for_json(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _maybe_date_like(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(40)
    if sample.empty:
        return False
    looks_date_like = sample.str.contains(
        r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2}|\d{4}-\d{2}-\d{2})",
        regex=True,
    ).mean()
    return bool(looks_date_like >= 0.5)


def _case_mismatch(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    normalized = sample.str.strip().str.lower()
    return normalized.nunique() < sample.nunique()


def _suspected_identifier(column_name: str, unique_ratio: float) -> bool:
    lowered = column_name.lower()
    return unique_ratio >= 0.85 and any(token in lowered for token in ("id", "uuid", "guid", "ticket", "record"))


def _column_profile(column_name: str, series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    row_count = len(series)
    missing_count = int(series.isna().sum())
    missing_ratio = float(missing_count / max(1, row_count))
    unique_count = int(non_null.nunique())
    unique_ratio = float(unique_count / max(1, len(non_null))) if len(non_null) else 0.0
    is_numeric = is_numeric_dtype(series)
    date_like = False if is_numeric else _maybe_date_like(series)
    constant = unique_count <= 1 and row_count > 0
    case_mismatch = False if is_numeric else _case_mismatch(series)
    identifier_like = _suspected_identifier(column_name, unique_ratio)
    outlier_fraction = 0.0
    if is_numeric and len(non_null) >= 8 and non_null.nunique() >= 4:
        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_fraction = float(((non_null < lower) | (non_null > upper)).mean())

    priority_score = (
        (missing_ratio * 5.0)
        + (outlier_fraction * 3.0)
        + (1.4 if identifier_like else 0.0)
        + (1.0 if constant else 0.0)
        + (0.9 if case_mismatch else 0.0)
        + (0.8 if date_like else 0.0)
        + (0.6 if unique_ratio > 0.9 and not identifier_like else 0.0)
    )

    workstreams: list[str] = []
    if missing_count > 0:
        workstreams.append("missing_value_repair")
    if case_mismatch:
        workstreams.append("categorical_normalization")
    if date_like:
        workstreams.append("datetime_engineering")
    if outlier_fraction > 0.02:
        workstreams.append("outlier_review")
    if identifier_like or constant:
        workstreams.append("schema_review")
    if unique_ratio > 0.9 and not is_numeric and not identifier_like:
        workstreams.append("high_cardinality_strategy")
    if not workstreams:
        workstreams.append("general_review")

    return {
        "column": column_name,
        "dtype": str(series.dtype),
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_ratio": round(missing_ratio, 4),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "constant": constant,
        "identifier_like": identifier_like,
        "date_like": date_like,
        "case_mismatch": case_mismatch,
        "outlier_fraction": round(outlier_fraction, 4),
        "priority_score": round(priority_score, 4),
        "suggested_workstreams": workstreams,
        "example_values": [_value_for_json(value) for value in non_null.head(3).tolist()],
    }


def build_dataset_profile(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    column_profiles = [_column_profile(column, df[column]) for column in df.columns]
    sorted_profiles = sorted(column_profiles, key=lambda item: item["priority_score"], reverse=True)
    missing_cells = int(df.isna().sum().sum())

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": missing_cells,
        "wide_dataset": len(df.columns) >= 40,
        "target_column": target_column,
        "column_profiles": column_profiles,
        "top_suspicious_columns": sorted_profiles[:MAX_TOP_COLUMNS],
    }


def build_work_queue(profile: dict[str, Any]) -> dict[str, Any]:
    profiles = profile.get("column_profiles", [])
    workstreams_map: dict[str, list[str]] = {}
    for column_profile in profiles:
        for workstream in column_profile["suggested_workstreams"]:
            workstreams_map.setdefault(workstream, []).append(column_profile["column"])

    workstream_metadata = {
        "missing_value_repair": ("high", "Repair missing values before any modeling decisions."),
        "categorical_normalization": ("medium", "Normalize labels and mixed-case categorical values."),
        "datetime_engineering": ("medium", "Expand date-like columns into modeling-friendly features."),
        "outlier_review": ("high", "Inspect numeric extremes before scaling or model fitting."),
        "schema_review": ("high", "Review identifiers, constants, and structurally suspicious columns."),
        "high_cardinality_strategy": ("medium", "Choose encoding strategies for wide categorical spaces."),
        "general_review": ("low", "Low-risk columns kept for final QA only."),
    }

    workstreams: list[dict[str, Any]] = []
    for name, columns in sorted(workstreams_map.items(), key=lambda item: (-len(item[1]), item[0])):
        priority, objective = workstream_metadata.get(name, ("medium", "Review these columns."))
        workstreams.append(
            {
                "workstream": name,
                "priority": priority,
                "column_count": len(columns),
                "columns": columns[:MAX_TOP_COLUMNS],
                "objective": objective,
            }
        )

    top_columns = [item["column"] for item in profile.get("top_suspicious_columns", [])]
    if not top_columns:
        top_columns = [item["column"] for item in profiles[:MAX_TOP_COLUMNS]]

    column_batches: list[dict[str, Any]] = []
    for index in range(0, len(top_columns), BATCH_SIZE):
        batch_columns = top_columns[index : index + BATCH_SIZE]
        column_batches.append(
            {
                "batch_id": f"batch_{(index // BATCH_SIZE) + 1}",
                "priority": "high" if index == 0 else "medium",
                "columns": batch_columns,
                "objective": "Review this subset before moving to the next wide-table batch.",
            }
        )

    recommendations = {
        "agent_strategy": (
            "Start with schema_review and missing_value_repair, then move through the batches of suspicious columns "
            "instead of attempting the entire dataset at once."
        ),
        "column_batch_size": BATCH_SIZE,
        "wide_dataset_mode": bool(profile.get("wide_dataset", False)),
    }
    return {
        "workstreams": workstreams,
        "column_batches": column_batches,
        "recommendations": recommendations,
    }


def _write_svg_bar_chart(items: list[tuple[str, float]], path: Path, title: str, accent: str = "#0f766e") -> None:
    width = 900
    height = max(220, 80 + (len(items) * 42))
    left_margin = 220
    right_margin = 40
    bar_area = width - left_margin - right_margin
    max_value = max((value for _, value in items), default=1.0) or 1.0

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="34" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#0f172a">{title}</text>',
    ]

    y = 70
    for label, value in items:
        bar_width = 0 if max_value == 0 else (float(value) / max_value) * bar_area
        safe_label = str(label)[:32]
        svg_lines.append(
            f'<text x="24" y="{y + 18}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#334155">{safe_label}</text>'
        )
        svg_lines.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_width:.1f}" height="22" rx="6" fill="{accent}" opacity="0.85"/>'
        )
        svg_lines.append(
            f'<text x="{left_margin + bar_width + 10:.1f}" y="{y + 17}" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#0f172a">{value:.4f}</text>'
        )
        y += 38

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_report_bundle(
    dataset_name: str,
    output_dir: str | Path,
    source_profile: dict[str, Any],
    prepared_profile: dict[str, Any],
    work_queue: dict[str, Any],
    preparation_summary: dict[str, Any],
    evaluation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    graphs_dir = reports_dir / "graphs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    top_missing = [
        (item["column"], float(item["missing_ratio"]))
        for item in sorted(source_profile["column_profiles"], key=lambda entry: entry["missing_ratio"], reverse=True)
        if item["missing_ratio"] > 0
    ][:MAX_TOP_COLUMNS]
    if not top_missing:
        top_missing = [("no_missing_columns", 0.0)]

    top_priority = [
        (item["column"], float(item["priority_score"]))
        for item in source_profile.get("top_suspicious_columns", [])[:MAX_TOP_COLUMNS]
    ]
    if not top_priority:
        top_priority = [("no_priority_flags", 0.0)]

    missing_chart = graphs_dir / f"{dataset_name}_top_missing_columns.svg"
    priority_chart = graphs_dir / f"{dataset_name}_top_priority_columns.svg"
    _write_svg_bar_chart(top_missing, missing_chart, "Top Missing Columns", accent="#2563eb")
    _write_svg_bar_chart(top_priority, priority_chart, "Highest Priority Columns", accent="#ca8a04")

    leaderboard_chart: Path | None = None
    if evaluation_summary and evaluation_summary.get("leaderboard"):
        leaderboard_chart = graphs_dir / f"{dataset_name}_model_leaderboard.svg"
        leaderboard_items = [
            (entry["model_name"], float(entry["primary_metric"]))
            for entry in evaluation_summary["leaderboard"][:MAX_TOP_COLUMNS]
        ]
        _write_svg_bar_chart(leaderboard_items, leaderboard_chart, "Model Leaderboard", accent="#059669")

    graph_paths = {
        "top_missing_columns": str(missing_chart),
        "top_priority_columns": str(priority_chart),
    }
    if leaderboard_chart is not None:
        graph_paths["model_leaderboard"] = str(leaderboard_chart)

    markdown_path = reports_dir / f"{dataset_name}_report.md"
    latex_path = reports_dir / f"{dataset_name}_report.tex"

    top_columns_rows = "\n".join(
        f"| {item['column']} | {item['dtype']} | {item['missing_ratio']:.2%} | {item['priority_score']:.2f} | {', '.join(item['suggested_workstreams'])} |"
        for item in source_profile.get("top_suspicious_columns", [])[:MAX_TOP_COLUMNS]
    ) or "| n/a | n/a | 0.00% | 0.00 | general_review |"

    workstream_rows = "\n".join(
        f"| {item['workstream']} | {item['priority']} | {item['column_count']} | {', '.join(item['columns'])} |"
        for item in work_queue.get("workstreams", [])
    ) or "| general_review | low | 0 | n/a |"

    evaluation_section = ""
    if evaluation_summary:
        leaderboard_rows = "\n".join(
            f"| {entry['model_name']} | {entry['primary_metric_name']} | {float(entry['primary_metric']):.4f} | {float(entry['training_seconds']):.2f}s |"
            for entry in evaluation_summary.get("leaderboard", [])
        )
        evaluation_section = (
            "## Evaluation Summary\n\n"
            f"- Best model: `{evaluation_summary['best_model']}`\n"
            f"- Primary metric: `{evaluation_summary['primary_metric_name']}`\n"
            f"- Best score: `{float(evaluation_summary['best_primary_metric']):.4f}`\n\n"
            "| Model | Metric | Score | Train Time |\n"
            "|---|---|---:|---:|\n"
            f"{leaderboard_rows}\n\n"
        )
        if leaderboard_chart is not None:
            evaluation_section += f"![Model leaderboard](graphs/{leaderboard_chart.name})\n\n"

    markdown = (
        f"# {dataset_name} data report\n\n"
        "## Dataset Overview\n\n"
        f"- Source rows: `{source_profile['rows']}`\n"
        f"- Source columns: `{source_profile['columns']}`\n"
        f"- Prepared rows: `{prepared_profile['rows']}`\n"
        f"- Prepared columns: `{prepared_profile['columns']}`\n"
        f"- Missing cells (source): `{source_profile['missing_cells']}`\n"
        f"- Duplicate rows (source): `{source_profile['duplicate_rows']}`\n"
        f"- Wide dataset mode: `{source_profile['wide_dataset']}`\n\n"
        "## Work Queue\n\n"
        "| Workstream | Priority | Column Count | Example Columns |\n"
        "|---|---|---:|---|\n"
        f"{workstream_rows}\n\n"
        "## Top Suspicious Columns\n\n"
        "| Column | DType | Missing Ratio | Priority Score | Suggested Workstreams |\n"
        "|---|---|---:|---:|---|\n"
        f"{top_columns_rows}\n\n"
        "## Agent Execution Strategy\n\n"
        f"- {work_queue['recommendations']['agent_strategy']}\n"
        f"- Recommended column batch size: `{work_queue['recommendations']['column_batch_size']}`\n"
        f"- Prepared feature count: `{preparation_summary.get('feature_count', prepared_profile['columns'])}`\n\n"
        "## Processing Steps\n\n"
        + "\n".join(f"- {step}" for step in preparation_summary.get("steps", []))
        + "\n\n"
        "## Charts\n\n"
        f"![Top missing columns](graphs/{missing_chart.name})\n\n"
        f"![Top priority columns](graphs/{priority_chart.name})\n\n"
        f"{evaluation_section}"
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    latex_workstreams = "\n".join(
        f"{_latex_escape(item['workstream'])} & {_latex_escape(item['priority'])} & {item['column_count']} \\\\"
        for item in work_queue.get("workstreams", [])[:MAX_TOP_COLUMNS]
    ) or r"general\_review & low & 0 \\"
    latex_top_columns = "\n".join(
        f"{_latex_escape(item['column'])} & {_latex_escape(item['dtype'])} & {item['missing_ratio']:.2f} & {item['priority_score']:.2f} \\\\"
        for item in source_profile.get("top_suspicious_columns", [])[:MAX_TOP_COLUMNS]
    ) or r"n/a & n/a & 0.00 & 0.00 \\"

    latex_evaluation = ""
    if evaluation_summary:
        leaderboard_rows = "\n".join(
            f"{_latex_escape(entry['model_name'])} & {_latex_escape(entry['primary_metric_name'])} & {float(entry['primary_metric']):.4f} \\\\"
            for entry in evaluation_summary.get("leaderboard", [])
        )
        latex_evaluation = (
            "\\section{Evaluation Summary}\n"
            f"Best model: \\texttt{{{_latex_escape(evaluation_summary['best_model'])}}}\\\\\n"
            f"Primary metric: \\texttt{{{_latex_escape(evaluation_summary['primary_metric_name'])}}}\\\\\n"
            f"Best score: {float(evaluation_summary['best_primary_metric']):.4f}\n\n"
            "\\begin{longtable}{lll}\n"
            "\\textbf{Model} & \\textbf{Metric} & \\textbf{Score} \\\\\n"
            "\\hline\n"
            f"{leaderboard_rows}\n"
            "\\end{longtable}\n\n"
        )
        if leaderboard_chart is not None:
            latex_evaluation += (
                "\\begin{figure}[h]\n\\centering\n"
                f"\\includesvg[width=0.95\\linewidth]{{graphs/{_latex_escape(leaderboard_chart.stem)}}}\n"
                "\\caption{Model leaderboard}\n\\end{figure}\n"
            )

    latex = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{svg}\n"
        "\\title{" + _latex_escape(f"{dataset_name} Data Report") + "}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section{Dataset Overview}\n"
        f"Source rows: {source_profile['rows']}\\\\\n"
        f"Source columns: {source_profile['columns']}\\\\\n"
        f"Prepared rows: {prepared_profile['rows']}\\\\\n"
        f"Prepared columns: {prepared_profile['columns']}\\\\\n"
        f"Missing cells (source): {source_profile['missing_cells']}\\\\\n"
        f"Duplicate rows (source): {source_profile['duplicate_rows']}\\\\\n"
        f"Wide dataset mode: {str(source_profile['wide_dataset']).lower()}\\\\\n\n"
        "\\section{Work Queue}\n"
        "\\begin{longtable}{lll}\n"
        "\\textbf{Workstream} & \\textbf{Priority} & \\textbf{Columns} \\\\\n"
        "\\hline\n"
        f"{latex_workstreams}\n"
        "\\end{longtable}\n\n"
        "\\section{Top Suspicious Columns}\n"
        "\\begin{longtable}{llll}\n"
        "\\textbf{Column} & \\textbf{DType} & \\textbf{Missing Ratio} & \\textbf{Priority} \\\\\n"
        "\\hline\n"
        f"{latex_top_columns}\n"
        "\\end{longtable}\n\n"
        "\\section{Processing Steps}\n"
        "\\begin{itemize}\n"
        + "\n".join(f"\\item {_latex_escape(step)}" for step in preparation_summary.get("steps", []))
        + "\n\\end{itemize}\n\n"
        "\\section{Charts}\n"
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includesvg[width=0.95\\linewidth]{{graphs/{_latex_escape(missing_chart.stem)}}}\n"
        "\\caption{Top missing columns}\n\\end{figure}\n"
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includesvg[width=0.95\\linewidth]{{graphs/{_latex_escape(priority_chart.stem)}}}\n"
        "\\caption{Highest priority columns}\n\\end{figure}\n\n"
        f"{latex_evaluation}"
        "\\end{document}\n"
    )
    latex_path.write_text(latex, encoding="utf-8")

    return {
        "markdown_report_path": str(markdown_path),
        "latex_report_path": str(latex_path),
        "graph_paths": graph_paths,
    }


def write_profile_bundle(
    dataset_name: str,
    output_dir: str | Path,
    source_profile: dict[str, Any],
    prepared_profile: dict[str, Any],
    work_queue: dict[str, Any],
) -> dict[str, str]:
    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile_path = reports_dir / f"{dataset_name}_profile.json"
    work_queue_path = reports_dir / f"{dataset_name}_work_queue.json"
    profile_path.write_text(
        json.dumps({"source_profile": source_profile, "prepared_profile": prepared_profile}, indent=2),
        encoding="utf-8",
    )
    work_queue_path.write_text(json.dumps(work_queue, indent=2), encoding="utf-8")
    return {
        "profile_path": str(profile_path),
        "work_queue_path": str(work_queue_path),
    }
