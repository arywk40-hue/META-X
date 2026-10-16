"""CLI entrypoint for generic dataset preparation."""

from __future__ import annotations

import argparse
import json

from environment.data_prep import apply_preprocessor, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a CSV dataset for direct model training.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", default=None, help="Optional target column name.")
    parser.add_argument("--output-dir", default=None, help="Directory for prepared artifacts.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction: 0.2 for an 80/20 split, 0.4 for 60/40.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--use-eda-agent",
        action="store_true",
        help="Run deterministic, schema-grounded EDA before preparation.",
    )
    parser.add_argument(
        "--eda-use-llm",
        action="store_true",
        help="Allow validated LLM suggestions on top of deterministic EDA when credentials are available.",
    )
    parser.add_argument(
        "--eda-llm-strategy",
        choices=["single_pass", "planner_reviewer"],
        default="single_pass",
        help="LLM orchestration mode for EDA suggestions.",
    )
    parser.add_argument(
        "--eda-llm-rounds",
        type=int,
        default=2,
        help="Bounded number of planner/reviewer rounds when using multi-agent EDA.",
    )
    parser.add_argument(
        "--task-type",
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Task type; auto infers it from the target column.",
    )
    parser.add_argument(
        "--mode",
        choices=["general", "credit_scoring"],
        default="general",
        help="credit_scoring requires a binary 0/1 target and selects models by roc_auc.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["auto", "stratified", "random", "group", "time"],
        default="auto",
        help="How to split rows into train and validation.",
    )
    parser.add_argument("--group-column", default=None, help="Group column; rows of one group stay in the same split.")
    parser.add_argument("--time-column", default=None, help="Timestamp column for chronological splits.")
    parser.add_argument(
        "--id-column",
        action="append",
        dest="id_columns",
        default=None,
        help="Identifier column kept alongside the output but never used as a feature. Repeat for more columns.",
    )
    parser.add_argument(
        "--exclude-column",
        action="append",
        dest="exclude_columns",
        default=None,
        help="Column removed from the features. Repeat for more columns.",
    )
    parser.add_argument(
        "--ratio-feature",
        action="append",
        dest="ratio_features",
        default=None,
        help="numerator:denominator; add only after confirming both columns are known at prediction time. Repeatable.",
    )
    parser.add_argument(
        "--apply-state",
        default=None,
        help="Saved preprocessor state JSON. Transforms --csv with it instead of preparing a dataset.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path for --apply-state.")
    args = parser.parse_args()

    if args.apply_state:
        if not args.output:
            parser.error("--output is required with --apply-state")
        result = apply_preprocessor(args.csv, args.apply_state, args.output)
        print(json.dumps(result, indent=2))
        return

    artifacts = prepare_dataset(
        csv_path=args.csv,
        target_column=args.target,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        use_eda_agent=args.use_eda_agent,
        eda_use_llm=args.eda_use_llm,
        eda_llm_strategy=args.eda_llm_strategy,
        eda_llm_rounds=args.eda_llm_rounds,
        task_type=args.task_type,
        mode=args.mode,
        split_strategy=args.split_strategy,
        group_column=args.group_column,
        time_column=args.time_column,
        id_columns=args.id_columns,
        exclude_columns=args.exclude_columns,
        ratio_features=args.ratio_features,
    )
    print(json.dumps(artifacts.as_dict(), indent=2))


if __name__ == "__main__":
    main()
