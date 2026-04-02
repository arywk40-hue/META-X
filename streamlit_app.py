"""Streamlit UI for generic dataset profiling, preparation, and evaluation."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from environment import Action, OpenEnv, generate_task_and_grader_from_csv
from environment.evaluation import prepare_and_evaluate_dataset
from environment.data_prep import prepare_dataset


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = APP_ROOT / "outputs" / "streamlit_uploads"
RUN_ROOT = APP_ROOT / "outputs" / "streamlit_runs"


def _ensure_dirs() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _save_upload(uploaded_file) -> Path:
    _ensure_dirs()
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    dataset_dir = UPLOAD_ROOT / f"{digest}_{uploaded_file.name.replace(' ', '_')}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / uploaded_file.name
    destination.write_bytes(file_bytes)
    return destination


def _profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing_counts = df.isna().sum().sort_values(ascending=False)
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_counts": missing_counts[missing_counts > 0],
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _load_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text()


def _download_button(path: str | None, label: str, mime: str = "text/plain") -> None:
    if not path:
        return
    file_path = Path(path)
    if not file_path.exists():
        return
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
        use_container_width=True,
    )


def _render_result_card(title: str, value: str, help_text: str | None = None) -> None:
    with st.container(border=True):
        st.caption(title)
        st.subheader(value)
        if help_text:
            st.caption(help_text)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _coerce_new_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if any(token in value for token in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _suggest_action_for_issue(issue: dict[str, Any]) -> dict[str, Any]:
    issue_type = str(issue["issue_type"])
    action_type = {
        "missing": "fill_missing",
        "null": "fill_missing",
        "string_null": "fix_value",
        "duplicate": "drop_row",
        "outlier": "flag_anomaly",
        "negative": "fix_value",
        "format": "standardize_format",
    }.get(issue_type, "fix_value")

    if issue_type == "duplicate":
        new_value = None
    elif issue.get("correct") is not None:
        new_value = issue["correct"]
    elif issue.get("allowed_values"):
        new_value = issue["allowed_values"][0]
    elif issue.get("value_type") == "numeric_range":
        minimum = float(issue["min"])
        maximum = float(issue["max"])
        new_value = round((minimum + maximum) / 2.0, 4)
    elif issue.get("value_type") == "iso_date":
        new_value = "2024-01-16"
    else:
        new_value = "fixed"

    return {
        "action_type": action_type,
        "row_index": int(issue["row_index"]),
        "column": str(issue["column"]),
        "new_value": new_value,
        "reason": f"Resolve {issue_type} issue.",
    }


def _dynamic_issue_rows(env: OpenEnv) -> list[dict[str, Any]]:
    task = env.custom_task
    if task is None:
        return []
    fixed = env.task_state.get("fixed_issues", set())
    rows: list[dict[str, Any]] = []
    for issue in task.config.get("issues", []):
        key = (int(issue["row_index"]), str(issue["column"]).strip())
        rows.append(
            {
                "resolved": key in fixed,
                "row_index": int(issue["row_index"]),
                "column": str(issue["column"]),
                "issue_type": str(issue["issue_type"]),
                "description": str(issue.get("description", "")),
            }
        )
    return rows


def _suggest_target_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred_names = [
        "target",
        "label",
        "class",
        "y",
        "outcome",
        "survived",
        "fraud",
        "response",
        "churn",
        "default",
        "is_fraud",
        "clicked",
        "price",
        "saleprice",
    ]
    lowered = {str(column).strip().lower(): str(column) for column in df.columns}
    for name in preferred_names:
        if name in lowered:
            return lowered[name]

    last_column = str(df.columns[-1])
    unique_count = int(df[last_column].nunique(dropna=True))
    if unique_count <= max(20, int(len(df) * 0.1)):
        return last_column
    return None


def _first_unresolved_issue_index(issue_rows: list[dict[str, Any]]) -> int:
    for index, issue in enumerate(issue_rows):
        if not bool(issue["resolved"]):
            return index
    return 0


def _run_dynamic_step(action_payload: dict[str, Any]) -> None:
    env: OpenEnv | None = st.session_state.get("metax_dynamic_env")
    if env is None:
        st.error("Generate a dynamic task first.")
        return
    try:
        observation, reward, done, info = env.step(Action.model_validate(action_payload))
    except Exception as exc:  # pragma: no cover - UI feedback surface
        st.error(f"Dynamic step failed: {exc}")
        return

    st.session_state["metax_dynamic_last_step"] = {
        "action": action_payload,
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


st.set_page_config(
    page_title="MetaX Data Prep Studio",
    page_icon="🧹",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root {
        --metax-bg: #040806;
        --metax-panel: #0b1510;
        --metax-panel-2: #101c15;
        --metax-border: rgba(46, 235, 127, 0.22);
        --metax-green: #2eeb7f;
        --metax-green-soft: #b8ffd2;
        --metax-text: #f5fff8;
        --metax-muted: #b4cbbd;
      }
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background:
          radial-gradient(circle at top left, rgba(46, 235, 127, 0.12), transparent 26%),
          radial-gradient(circle at top right, rgba(120, 255, 186, 0.08), transparent 22%),
          linear-gradient(180deg, #020503 0%, #08100c 46%, #040806 100%);
        color: var(--metax-text);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08110c 0%, #060d09 100%);
        border-right: 1px solid rgba(46, 235, 127, 0.12);
      }
      [data-testid="stSidebar"] * {
        color: var(--metax-text);
      }
      .metax-hero {
        padding: 1.25rem 1.4rem;
        border-radius: 20px;
        background:
          linear-gradient(135deg, rgba(3, 8, 5, 0.98) 0%, rgba(7, 18, 11, 0.96) 44%, rgba(22, 92, 48, 0.94) 100%);
        color: var(--metax-text);
        border: 1px solid rgba(46, 235, 127, 0.18);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.42);
      }
      .metax-hero h1 {margin: 0 0 0.2rem 0; color: #ffffff;}
      .metax-hero p {margin: 0.2rem 0 0 0; color: var(--metax-green-soft);}
      .small-note {color: var(--metax-muted); font-size: 0.92rem;}
      h1, h2, h3, h4, h5, h6, p, li, label, span, div, small {
        color: var(--metax-text);
      }
      .stCaption, [data-testid="stCaptionContainer"], .stMarkdown, .stText, .stAlert {
        color: var(--metax-text);
      }
      [data-testid="stMetric"], [data-testid="stMetric"] * {
        color: var(--metax-text) !important;
      }
      [data-testid="stVerticalBlockBorderWrapper"],
      [data-testid="stExpander"],
      [data-testid="stForm"],
      .stDataFrame,
      [data-testid="stCodeBlock"] {
        background: linear-gradient(180deg, rgba(11, 21, 16, 0.92) 0%, rgba(7, 13, 10, 0.96) 100%);
        border: 1px solid var(--metax-border);
        border-radius: 16px;
      }
      [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
      }
      [data-testid="stDataFrame"] [role="grid"] {
        background: rgba(8, 16, 12, 0.95);
        color: var(--metax-text);
      }
      [data-testid="stDataFrame"] [role="columnheader"],
      [data-testid="stDataFrame"] [role="gridcell"] {
        color: var(--metax-text) !important;
        background: transparent !important;
      }
      [data-testid="stDataFrame"] [role="columnheader"] {
        background: rgba(19, 37, 27, 0.95) !important;
      }
      .stSelectbox label,
      .stTextInput label,
      .stTextArea label,
      .stNumberInput label,
      .stSlider label,
      .stFileUploader label,
      .stCheckbox label {
        color: var(--metax-green-soft) !important;
      }
      .stTextInput input,
      .stNumberInput input,
      .stTextArea textarea,
      div[data-baseweb="select"] > div,
      div[data-baseweb="base-input"] {
        background: rgba(12, 22, 17, 0.96) !important;
        color: var(--metax-text) !important;
        border: 1px solid rgba(46, 235, 127, 0.24) !important;
      }
      div[data-baseweb="select"] svg {
        fill: var(--metax-green-soft);
      }
      .stButton button,
      .stDownloadButton button,
      [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #0c1c13 0%, #154b2a 100%);
        color: #f7fff9;
        border: 1px solid rgba(46, 235, 127, 0.4);
        border-radius: 12px;
      }
      .stButton button:hover,
      .stDownloadButton button:hover,
      [data-testid="stFormSubmitButton"] button:hover {
        border-color: rgba(46, 235, 127, 0.7);
        color: #ffffff;
        box-shadow: 0 0 0 1px rgba(46, 235, 127, 0.18), 0 10px 20px rgba(0, 0, 0, 0.28);
      }
      .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #14984b 0%, #2eeb7f 100%);
        color: #031007;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
      }
      .stTabs [data-baseweb="tab"] {
        background: rgba(10, 19, 14, 0.92);
        border: 1px solid rgba(46, 235, 127, 0.12);
        border-radius: 12px;
        color: var(--metax-text);
      }
      .stTabs [aria-selected="true"] {
        background: rgba(25, 74, 43, 0.95) !important;
        border-color: rgba(46, 235, 127, 0.5) !important;
        color: #ffffff !important;
      }
      .stAlert {
        background: rgba(9, 18, 13, 0.94);
        border: 1px solid rgba(46, 235, 127, 0.18);
        border-radius: 14px;
      }
      code, pre {
        color: #d7ffe6 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metax-hero">
      <h1>MetaX OpenEnv Data Cleaning</h1>
      <p>Benchmark-first OpenEnv environment for tabular data cleaning, with an optional studio layer for arbitrary CSV profiling, feature engineering, and evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='small-note'>Core submission: fixed OpenEnv benchmark tasks with rewards and graders. Demo extension: tabular CSV datasets for classification or regression through the studio workflow.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin profiling and preparation.")
    st.stop()

csv_path = _save_upload(uploaded)
df = pd.read_csv(csv_path)
profile = _profile_dataframe(df)
if st.session_state.get("metax_uploaded_csv_path") != str(csv_path):
    st.session_state["metax_uploaded_csv_path"] = str(csv_path)
    st.session_state["metax_target_column"] = _suggest_target_column(df) or ""
    st.session_state.pop("metax_dynamic_env", None)
    st.session_state.pop("metax_dynamic_last_step", None)
    st.session_state.pop("metax_dynamic_issue_index", None)

suggested_target = _suggest_target_column(df)

tab_inspect, tab_run, tab_results, tab_dynamic = st.tabs(
    ["1. Inspect Dataset", "2. Configure & Run", "3. Results", "4. Dynamic OpenEnv"]
)

with tab_inspect:
    left, right = st.columns([1.3, 1.0], gap="large")
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        with st.expander("Column names", expanded=False):
            st.write(df.columns.tolist())

    with right:
        st.subheader("Quick Profile")
        top_row = st.columns(3)
        with top_row[0]:
            _render_result_card("Rows", str(profile["rows"]))
        with top_row[1]:
            _render_result_card("Columns", str(profile["columns"]))
        with top_row[2]:
            _render_result_card("Duplicate Rows", str(profile["duplicate_rows"]))
        _render_result_card("Suggested Target", suggested_target or "Not detected")
        with st.expander("Column categories", expanded=False):
            st.write(
                {
                    "numeric": profile["numeric_columns"],
                    "categorical": profile["categorical_columns"],
                }
            )

    if not profile["missing_counts"].empty:
        st.subheader("Missing Values")
        missing_df = profile["missing_counts"].reset_index()
        missing_df.columns = ["column", "missing_count"]
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.info("No missing values detected in the uploaded sample.")

with tab_run:
    st.subheader("Configuration")
    st.caption("Step 1: choose a target, Step 2: decide how much evaluation you want, Step 3: run the pipeline.")

    config_cols = st.columns([1.0, 1.0, 1.0, 1.0], gap="medium")
    with config_cols[0]:
        target_column = st.selectbox(
            "Target column",
            options=[""] + df.columns.tolist(),
            key="metax_target_column",
        )
    with config_cols[1]:
        validation_fraction = st.slider("Validation split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with config_cols[2]:
        run_mode = st.selectbox("Mode", options=["Prepare only", "Prepare and evaluate"])
    with config_cols[3]:
        run_label = st.text_input("Run label", value="latest_run")

    if suggested_target and target_column != suggested_target:
        st.warning(f"Suggested target for this dataset: `{suggested_target}`. Right now you selected `{target_column or 'none'}`.")

    eda_cols = st.columns([1.0, 1.0, 1.0, 1.0], gap="medium")
    with eda_cols[0]:
        use_eda_agent = st.checkbox(
            "Enable schema-grounded EDA + feature engineering",
            value=True,
            help="Always runs deterministic profiling first, then prepares feature-engineering steps grounded in the actual dataset schema.",
        )
    with eda_cols[1]:
        eda_use_llm = st.checkbox(
            "Enhance with validated LLM suggestions",
            value=False,
            help="Optional. The LLM only proposes extra ideas on top of deterministic EDA, and unsafe or schema-breaking suggestions are rejected before they touch the dataset.",
            disabled=not use_eda_agent,
        )
    with eda_cols[2]:
        eda_llm_strategy = st.selectbox(
            "LLM orchestration",
            options=["single_pass", "planner_reviewer"],
            format_func=lambda value: "Single-pass assist" if value == "single_pass" else "Planner + reviewer loop",
            index=1,
            disabled=not (use_eda_agent and eda_use_llm),
            help="Planner + reviewer runs a bounded two-agent loop: one model proposes, another critiques, and only validated steps survive.",
        )
    with eda_cols[3]:
        eda_llm_rounds = st.slider(
            "LLM rounds",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            disabled=not (use_eda_agent and eda_use_llm and eda_llm_strategy == "planner_reviewer"),
            help="Bounded rounds keep runtime predictable while still letting the models refine ideas.",
        )

    with st.expander("What happens when you run", expanded=True):
        st.markdown(
            """
1. The dataset is profiled and checked for obvious quality issues.
2. Optional EDA + feature-engineering steps are prepared, either in one pass or through a bounded planner + reviewer loop.
3. The CSV is cleaned into train-ready features.
4. If you choose `Prepare and evaluate`, fast baseline models are trained on the validation split.
            """
        )

    run_dir = RUN_ROOT / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{run_label.strip().replace(' ', '_')}"
    if st.button("Run pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing dataset..."):
            if run_mode == "Prepare only":
                result = prepare_dataset(
                    csv_path=str(csv_path),
                    target_column=target_column or None,
                    output_dir=str(run_dir),
                    validation_fraction=validation_fraction,
                    use_eda_agent=use_eda_agent,
                    eda_use_llm=eda_use_llm,
                    eda_llm_strategy=eda_llm_strategy,
                    eda_llm_rounds=eda_llm_rounds,
                ).as_dict()
                st.session_state["metax_result"] = {"preparation": result, "evaluation": None}
            else:
                if not target_column:
                    st.error("Choose a target column for prepare-and-evaluate mode.")
                    st.stop()
                result = prepare_and_evaluate_dataset(
                    csv_path=str(csv_path),
                    target_column=target_column,
                    output_dir=str(run_dir),
                    validation_fraction=validation_fraction,
                    use_eda_agent=use_eda_agent,
                    eda_use_llm=eda_use_llm,
                    eda_llm_strategy=eda_llm_strategy,
                    eda_llm_rounds=eda_llm_rounds,
                )
                st.session_state["metax_result"] = result
        st.success("Pipeline complete. Open the Results tab to inspect artifacts and metrics.")

with tab_results:
    result = st.session_state.get("metax_result")
    if not result:
        st.info("Run the pipeline from the Configure & Run tab to see outputs here.")
    else:
        prep = result["preparation"]
        eval_result = result.get("evaluation")

        st.subheader("Preparation Summary")
        summary_cols = st.columns(4)
        with summary_cols[0]:
            _render_result_card("Prepared Rows", str(prep["prepared_shape"]["rows"]))
        with summary_cols[1]:
            _render_result_card("Prepared Columns", str(prep["prepared_shape"]["columns"]))
        with summary_cols[2]:
            _render_result_card("Feature Count", str(prep["feature_count"]))
        with summary_cols[3]:
            _render_result_card("Task Type", prep["task_type"] or "unknown")

        overview_cols = st.columns(2)
        with overview_cols[0]:
            with st.expander("Processing steps", expanded=True):
                st.write(prep["steps"])
        with overview_cols[1]:
            with st.expander("Feature columns", expanded=False):
                st.write(prep["feature_columns"])

        downloads = st.columns(4)
        with downloads[0]:
            _download_button(prep.get("prepared_full_path"), "Download Full Prepared CSV", mime="text/csv")
        with downloads[1]:
            _download_button(prep.get("prepared_train_path"), "Download Train CSV", mime="text/csv")
        with downloads[2]:
            _download_button(prep.get("prepared_valid_path"), "Download Valid CSV", mime="text/csv")
        with downloads[3]:
            _download_button(prep.get("manifest_path"), "Download Feature Manifest", mime="application/json")

        report_downloads = st.columns(4)
        with report_downloads[0]:
            _download_button(prep.get("profile_path"), "Download Profile JSON", mime="application/json")
        with report_downloads[1]:
            _download_button(prep.get("work_queue_path"), "Download Work Queue JSON", mime="application/json")
        with report_downloads[2]:
            _download_button(prep.get("markdown_report_path"), "Download Markdown Report", mime="text/markdown")
        with report_downloads[3]:
            _download_button(prep.get("latex_report_path"), "Download LaTeX Report", mime="text/plain")

        if prep.get("eda_enabled"):
            st.subheader("EDA + Feature Engineering")
            eda_cards = st.columns(5)
            with eda_cards[0]:
                _render_result_card("LLM Used", str(prep.get("eda_used_llm", False)))
            with eda_cards[1]:
                _render_result_card("LLM Strategy", str(prep.get("eda_llm_strategy", "none")))
            with eda_cards[2]:
                _render_result_card("Rounds Run", str(prep.get("eda_llm_rounds_run", 0)))
            with eda_cards[3]:
                _render_result_card("Candidate LLM Steps", str(prep.get("eda_llm_candidate_steps", 0)))
            with eda_cards[4]:
                _render_result_card("Feature Steps", str(prep.get("eda_feature_engineering_steps", 0)))
            step_cards = st.columns(3)
            with step_cards[0]:
                _render_result_card("Validated LLM Steps", str(prep.get("eda_validated_llm_steps", 0)))
            with step_cards[1]:
                _render_result_card("Rejected LLM Steps", str(prep.get("eda_rejected_llm_steps", 0)))
            with step_cards[2]:
                _render_result_card("LLM Provider", str(prep.get("eda_llm_provider", "none")))
            with st.expander("EDA summary and recommendations", expanded=True):
                st.write(
                    {
                        "summary": prep.get("eda_summary"),
                        "recommendations": prep.get("eda_recommendations", []),
                    }
                )
            if prep.get("eda_llm_round_records"):
                with st.expander("Planner / reviewer dialogue", expanded=False):
                    for round_record in prep.get("eda_llm_round_records", []):
                        st.markdown(f"**Round {round_record['round_index']}**")
                        st.write(
                            {
                                "planner_summary": round_record["planner_summary"],
                                "planner_recommendations": round_record["planner_recommendations"],
                                "planner_step_count": round_record["planner_step_count"],
                                "reviewer_summary": round_record["reviewer_summary"],
                                "reviewer_recommendations": round_record["reviewer_recommendations"],
                                "reviewer_step_count": round_record["reviewer_step_count"],
                                "accepted_step_count": round_record["accepted_step_count"],
                                "rejected_step_count": round_record["rejected_step_count"],
                            }
                        )
            with st.expander("Apply EDA result to the dataset", expanded=True):
                st.caption(
                    "These validated EDA transformations were already applied once before the main preparation pipeline. "
                    "You can download that intermediate CSV directly here."
                )
                eda_apply_cards = st.columns(3)
                with eda_apply_cards[0]:
                    shape = prep.get("eda_applied_shape") or {}
                    rows_text = str(shape.get("rows", "N/A"))
                    _render_result_card("EDA Rows", rows_text)
                with eda_apply_cards[1]:
                    shape = prep.get("eda_applied_shape") or {}
                    cols_text = str(shape.get("columns", "N/A"))
                    _render_result_card("EDA Columns", cols_text)
                with eda_apply_cards[2]:
                    _render_result_card("Added Columns", str(len(prep.get("eda_added_columns", []))))
                _download_button(prep.get("eda_applied_path"), "Download EDA-Applied CSV", mime="text/csv")
                st.write(
                    {
                        "eda_added_columns": prep.get("eda_added_columns", []),
                        "eda_removed_columns": prep.get("eda_removed_columns", []),
                    }
                )
            if prep.get("eda_llm_rejection_reasons"):
                with st.expander("Why LLM steps were rejected", expanded=False):
                    st.write(prep.get("eda_llm_rejection_reasons", []))
            eda_downloads = st.columns(2)
            with eda_downloads[0]:
                _download_button(prep.get("eda_report_path"), "Download EDA Report JSON", mime="application/json")
            with eda_downloads[1]:
                _download_button(prep.get("eda_markdown_path"), "Download EDA Markdown", mime="text/markdown")

        if eval_result:
            st.subheader("Evaluation Summary")
            if suggested_target and prep.get("target_column") != suggested_target:
                st.warning(
                    f"You evaluated with target `{prep.get('target_column')}`, but this dataset looks like it should use `{suggested_target}`. "
                    "That can change the task type and hide the metric you expected, like ROC-AUC."
                )

            leaderboard = eval_result["leaderboard"]
            best_entry = leaderboard[0] if leaderboard else {}

            eval_cards = st.columns(3)
            with eval_cards[0]:
                _render_result_card("Best Model", eval_result["best_model"])
            with eval_cards[1]:
                _render_result_card("Primary Metric", eval_result["primary_metric_name"])
            with eval_cards[2]:
                _render_result_card("Best Score", f"{eval_result['best_primary_metric']:.6f}")

            metric_name = str(eval_result["primary_metric_name"])
            if metric_name == "roc_auc" or prep.get("task_type") == "classification":
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    _render_result_card("ROC-AUC", _format_metric_value(best_entry.get("roc_auc")))
                with metric_cols[1]:
                    _render_result_card("Accuracy", _format_metric_value(best_entry.get("accuracy")))
                with metric_cols[2]:
                    _render_result_card("F1 Macro", _format_metric_value(best_entry.get("f1_macro")))
                with metric_cols[3]:
                    _render_result_card("Log Loss", _format_metric_value(best_entry.get("log_loss")))
            else:
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    _render_result_card("R2 Score", _format_metric_value(best_entry.get("r2")))
                with metric_cols[1]:
                    _render_result_card("RMSE", _format_metric_value(best_entry.get("rmse")))
                with metric_cols[2]:
                    _render_result_card("MAE", _format_metric_value(best_entry.get("mae")))
                with metric_cols[3]:
                    _render_result_card("Training Sec", _format_metric_value(best_entry.get("training_seconds")))

            leaderboard_df = pd.DataFrame(leaderboard)
            st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

            eval_downloads = st.columns(3)
            with eval_downloads[0]:
                _download_button(eval_result.get("evaluation_report_path"), "Download Evaluation Report", mime="application/json")
            with eval_downloads[1]:
                _download_button(eval_result.get("markdown_report_path"), "Download Final Markdown Report", mime="text/markdown")
            with eval_downloads[2]:
                _download_button(eval_result.get("latex_report_path"), "Download Final LaTeX Report", mime="text/plain")
            with st.expander("Evaluation report JSON", expanded=False):
                report_path = eval_result.get("evaluation_report_path")
                if report_path:
                    st.code(_load_text(report_path), language="json")

with tab_dynamic:
    st.subheader("Dynamic OpenEnv Task")
    st.caption(
        "Generate a session-local RL cleaning episode from this uploaded CSV and work through issues one by one."
    )

    if st.session_state.get("metax_dynamic_csv_path") not in {None, str(csv_path)}:
        st.session_state.pop("metax_dynamic_env", None)
        st.session_state.pop("metax_dynamic_last_step", None)
        st.session_state["metax_dynamic_csv_path"] = str(csv_path)

    dynamic_cols = st.columns([1.0, 1.0, 1.0], gap="medium")
    with dynamic_cols[0]:
        max_dynamic_issues = st.slider("Max detected issues", min_value=3, max_value=12, value=7, step=1)
    with dynamic_cols[1]:
        dynamic_preview_rows = st.slider("Preview rows in RL task", min_value=5, max_value=20, value=10, step=1)
    with dynamic_cols[2]:
        dynamic_task_id = st.text_input("Optional dynamic task id", value="")

    dynamic_action_cols = st.columns([1.1, 0.9], gap="medium")
    with dynamic_action_cols[0]:
        if st.button("Generate dynamic OpenEnv task", use_container_width=True):
            try:
                task, grader = generate_task_and_grader_from_csv(
                    str(csv_path),
                    task_id=dynamic_task_id or None,
                    max_issues=max_dynamic_issues,
                    max_preview_rows=dynamic_preview_rows,
                )
                env = OpenEnv()
                env.set_dynamic_task(task, grader)
                env.reset(task.id)
                st.session_state["metax_dynamic_env"] = env
                st.session_state["metax_dynamic_csv_path"] = str(csv_path)
                st.session_state["metax_dynamic_last_step"] = None
                st.session_state["metax_dynamic_issue_index"] = 0
                st.success(f"Generated dynamic task `{task.id}` with {len(task.config['issues'])} detected issues.")
            except Exception as exc:  # pragma: no cover - UI feedback surface
                st.error(f"Could not generate dynamic task: {exc}")

    with dynamic_action_cols[1]:
        if st.button("Reset dynamic episode", use_container_width=True):
            env = st.session_state.get("metax_dynamic_env")
            if env is not None and env.custom_task is not None:
                env.reset(env.custom_task.id)
                st.session_state["metax_dynamic_last_step"] = None
                st.session_state["metax_dynamic_issue_index"] = 0
                st.success("Dynamic episode reset.")

    dynamic_env: OpenEnv | None = st.session_state.get("metax_dynamic_env")
    if dynamic_env is None or dynamic_env.custom_task is None or dynamic_env.current_observation is None:
        st.info("Generate a dynamic task to start the one-by-one RL workflow.")
    else:
        dynamic_task = dynamic_env.custom_task
        dynamic_observation = dynamic_env.current_observation
        dynamic_state = dynamic_env.state()
        dynamic_issue_rows = _dynamic_issue_rows(dynamic_env)
        issue_df = pd.DataFrame(dynamic_issue_rows)
        unresolved_count = int((~issue_df["resolved"]).sum()) if not issue_df.empty else 0

        top_cards = st.columns(4)
        with top_cards[0]:
            _render_result_card("Dynamic Task", dynamic_task.id)
        with top_cards[1]:
            _render_result_card("Difficulty", dynamic_task.difficulty.value)
        with top_cards[2]:
            _render_result_card("Issues Remaining", str(dynamic_observation.issues_remaining))
        with top_cards[3]:
            _render_result_card("Attempts Left", str(dynamic_observation.attempts_remaining))

        if dynamic_issue_rows:
            default_issue_index = min(
                st.session_state.get("metax_dynamic_issue_index", _first_unresolved_issue_index(dynamic_issue_rows)),
                len(dynamic_issue_rows) - 1,
            )
            selected_index = st.selectbox(
                "Step 1: pick the next issue to inspect",
                options=list(range(len(dynamic_issue_rows))),
                index=default_issue_index,
                format_func=lambda idx: (
                    f"{'done' if dynamic_issue_rows[idx]['resolved'] else 'todo'} | "
                    f"row {dynamic_issue_rows[idx]['row_index']} | "
                    f"{dynamic_issue_rows[idx]['column']} | "
                    f"{dynamic_issue_rows[idx]['issue_type']}"
                ),
                key="metax_dynamic_issue_index",
            )
            selected_issue = dynamic_task.config["issues"][selected_index]
            suggested_action = _suggest_action_for_issue(selected_issue)
        else:
            selected_issue = None
            suggested_action = None

        detail_cols = st.columns([1.05, 0.95], gap="large")
        with detail_cols[0]:
            if selected_issue is not None:
                st.subheader("Selected Issue")
                st.write(
                    {
                        "row_index": selected_issue["row_index"],
                        "column": selected_issue["column"],
                        "issue_type": selected_issue["issue_type"],
                        "description": selected_issue.get("description", ""),
                    }
                )
            with st.expander("Current observation preview", expanded=True):
                st.code(dynamic_observation.dataset_preview, language="text")
                st.caption(dynamic_observation.feedback or "No feedback yet.")
            with st.expander("All detected issues", expanded=False):
                st.dataframe(issue_df, use_container_width=True, hide_index=True)

        with detail_cols[1]:
            if suggested_action is not None:
                st.subheader("Step 2: review suggested action")
                st.json(suggested_action)
                if st.button("Apply suggested action", use_container_width=True):
                    _run_dynamic_step(suggested_action)

            st.subheader("Step 3: or apply your own action")
            with st.form("metax_dynamic_manual_action"):
                action_options = Action.model_json_schema()["properties"]["action_type"]["enum"]
                default_action = suggested_action["action_type"] if suggested_action else action_options[0]
                default_index = action_options.index(default_action)
                action_type = st.selectbox("Action type", action_options, index=default_index)
                row_index = st.number_input(
                    "Row index",
                    min_value=0,
                    value=int(suggested_action["row_index"]) if suggested_action else 0,
                    step=1,
                )
                column = st.text_input("Column", value=str(suggested_action["column"]) if suggested_action else "")
                new_value = st.text_input(
                    "New value",
                    value="" if not suggested_action or suggested_action["new_value"] is None else str(suggested_action["new_value"]),
                )
                reason = st.text_area(
                    "Reason",
                    value=str(suggested_action["reason"]) if suggested_action else "Manual dynamic action.",
                    height=90,
                )
                manual_submit = st.form_submit_button("Apply manual action", use_container_width=True)

                if manual_submit:
                    _run_dynamic_step(
                        {
                            "action_type": action_type,
                            "row_index": int(row_index),
                            "column": column,
                            "new_value": _coerce_new_value(new_value),
                            "reason": reason,
                        }
                    )

        last_dynamic_step = st.session_state.get("metax_dynamic_last_step")
        if last_dynamic_step:
            st.subheader("Last Step Result")
            step_cards = st.columns(4)
            with step_cards[0]:
                _render_result_card("Reward", f"{last_dynamic_step['reward']['value']:.4f}")
            with step_cards[1]:
                _render_result_card("Done", str(last_dynamic_step["done"]))
            with step_cards[2]:
                _render_result_card("Issues Fixed", str(last_dynamic_step["info"]["issues_fixed_this_step"]))
            with step_cards[3]:
                _render_result_card("Unresolved Issues", str(unresolved_count))
            with st.expander("Raw last-step JSON", expanded=False):
                st.json(last_dynamic_step)

        task_downloads = st.columns(2)
        with task_downloads[0]:
            st.download_button(
                "Download Dynamic Task JSON",
                data=json.dumps(
                    {
                        **dynamic_task.summary(),
                        "action_schema": Action.model_json_schema(),
                        "issues": dynamic_task.config.get("issues", []),
                    },
                    indent=2,
                ),
                file_name=f"{dynamic_task.id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with task_downloads[1]:
            st.download_button(
                "Download Current Dynamic State",
                data=json.dumps(dynamic_state, indent=2),
                file_name=f"{dynamic_task.id}_state.json",
                mime="application/json",
                use_container_width=True,
            )
