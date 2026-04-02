"""CLI entrypoint for generic dataset preparation."""

from __future__ import annotations

import argparse
import json

from environment.data_prep import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a CSV dataset for direct model training.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", default=None, help="Optional target column name.")
    parser.add_argument("--output-dir", default=None, help="Directory for prepared artifacts.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Validation split fraction.")
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
    args = parser.parse_args()

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
    )
    print(json.dumps(artifacts.as_dict(), indent=2))


if __name__ == "__main__":
    main()
