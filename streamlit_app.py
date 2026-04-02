"""Streamlit UI for generic dataset profiling, preparation, and evaluation."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_dir = UPLOAD_ROOT / f"{timestamp}_{uploaded_file.name.replace(' ', '_')}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
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


st.set_page_config(
    page_title="MetaX Data Prep Studio",
    page_icon="🧹",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(17, 110, 255, 0.10), transparent 28%),
          radial-gradient(circle at top right, rgba(14, 159, 110, 0.10), transparent 24%),
          linear-gradient(180deg, #f6f8fb 0%, #eef3f7 100%);
      }
      .metax-hero {
        padding: 1.25rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #153e75 55%, #166534 100%);
        color: #f8fafc;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
      }
      .metax-hero h1 {margin: 0 0 0.2rem 0; color: #ffffff;}
      .metax-hero p {margin: 0.2rem 0 0 0; color: #dbeafe;}
      .small-note {color: #475569; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metax-hero">
      <h1>MetaX Data Prep Studio</h1>
      <p>Upload any tabular CSV, inspect the schema, generate train-ready features, and benchmark fast baseline models in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='small-note'>Supported today: tabular CSV datasets for classification or regression. Image, text, and time-series specific modeling are not yet specialized.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin profiling and preparation.")
    st.stop()

csv_path = _save_upload(uploaded)
df = pd.read_csv(csv_path)
profile = _profile_dataframe(df)

left, right = st.columns([1.3, 1.0], gap="large")
with left:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

with right:
    st.subheader("Quick Profile")
    top_row = st.columns(3)
    with top_row[0]:
        _render_result_card("Rows", str(profile["rows"]))
    with top_row[1]:
        _render_result_card("Columns", str(profile["columns"]))
    with top_row[2]:
        _render_result_card("Duplicate Rows", str(profile["duplicate_rows"]))

    st.caption("Column categories")
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

st.subheader("Configuration")
config_cols = st.columns([1.0, 1.0, 1.0, 1.0], gap="medium")
with config_cols[0]:
    target_column = st.selectbox("Target column", options=[""] + df.columns.tolist(), index=0)
with config_cols[1]:
    validation_fraction = st.slider("Validation split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
with config_cols[2]:
    run_mode = st.selectbox("Mode", options=["Prepare only", "Prepare and evaluate"])
with config_cols[3]:
    run_label = st.text_input("Run label", value=_save_upload.__name__.replace("_save_upload", "latest_run"))

eda_cols = st.columns([1.0, 1.0], gap="medium")
with eda_cols[0]:
    use_eda_agent = st.checkbox(
        "Enable EDA agent",
        value=True,
        help="Run schema-aware EDA first so the pipeline can handle new columns and unfamiliar datasets more robustly.",
    )
with eda_cols[1]:
    eda_use_llm = st.checkbox(
        "Use LLM for EDA agent",
        value=False,
        help="Requires API credentials. When off, the EDA agent still runs with statistical heuristics.",
        disabled=not use_eda_agent,
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
            )
            st.session_state["metax_result"] = result

result = st.session_state.get("metax_result")
if result:
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

    st.write("Feature columns", prep["feature_columns"])
    st.write("Processing steps", prep["steps"])

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
        st.subheader("EDA Agent")
        st.write(
            {
                "used_llm": prep.get("eda_used_llm", False),
                "feature_steps": prep.get("eda_feature_engineering_steps", 0),
                "summary": prep.get("eda_summary"),
                "recommendations": prep.get("eda_recommendations", []),
            }
        )
        eda_downloads = st.columns(2)
        with eda_downloads[0]:
            _download_button(prep.get("eda_report_path"), "Download EDA Report JSON", mime="application/json")
        with eda_downloads[1]:
            _download_button(prep.get("eda_markdown_path"), "Download EDA Markdown", mime="text/markdown")

    if eval_result:
        st.subheader("Evaluation Summary")
        eval_cards = st.columns(3)
        with eval_cards[0]:
            _render_result_card("Best Model", eval_result["best_model"])
        with eval_cards[1]:
            _render_result_card("Primary Metric", eval_result["primary_metric_name"])
        with eval_cards[2]:
            _render_result_card("Best Score", f"{eval_result['best_primary_metric']:.6f}")

        leaderboard_df = pd.DataFrame(eval_result["leaderboard"])
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        report_path = eval_result.get("evaluation_report_path")
        eval_downloads = st.columns(3)
        with eval_downloads[0]:
            _download_button(report_path, "Download Evaluation Report", mime="application/json")
        with eval_downloads[1]:
            _download_button(eval_result.get("markdown_report_path"), "Download Final Markdown Report", mime="text/markdown")
        with eval_downloads[2]:
            _download_button(eval_result.get("latex_report_path"), "Download Final LaTeX Report", mime="text/plain")
        if report_path:
            with st.expander("Evaluation Report JSON"):
                st.code(_load_text(report_path), language="json")
