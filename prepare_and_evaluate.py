"""CLI entrypoint for end-to-end dataset preparation and validation scoring."""

from __future__ import annotations

import argparse
import json

from environment.evaluation import prepare_and_evaluate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a CSV dataset and evaluate fast baseline models.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output-dir", default=None, help="Directory for prepared artifacts and evaluation report.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    args = parser.parse_args()

    result = prepare_and_evaluate_dataset(
        csv_path=args.csv,
        target_column=args.target,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
