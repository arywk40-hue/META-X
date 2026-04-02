"""Convenience wrapper for the dataset EDA agent."""

from environment.eda_agent import (
    ColumnProfile,
    CorrelationInsight,
    EDAAgent,
    EDAReport,
    FeatureEngineeringStep,
    LLMRoundRecord,
    main,
    write_eda_artifacts,
)

__all__ = [
    "ColumnProfile",
    "CorrelationInsight",
    "EDAAgent",
    "EDAReport",
    "FeatureEngineeringStep",
    "LLMRoundRecord",
    "write_eda_artifacts",
]


if __name__ == "__main__":
    main()
