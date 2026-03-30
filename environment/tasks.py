"""Task registry for the data-cleaning environment."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import Difficulty, Task


EASY_DATASET = """\
row_id | name          | age  | email
-------|---------------|------|----------------------
0      | Alice Chen    | 29   | alice@example.com
1      | Bob Marsh     |      | bob@example.com
2      | Carol Diaz    | NULL | carol@example.com
3      | Dave Kim      | 31   | dave@example.com
4      | Eve Santos    | 27   |
"""

MEDIUM_DATASET = """\
row_id | date            | revenue | currency
-------|-----------------|---------|----------
0      | 2024-01-15      | 1500    | USD
1      | 15/01/2024      | 1200    | usd
2      | Jan 15 2024     | 980     | USD
3      | 2024-01-15      | 1500    | US Dollar
4      | 01-15-2024      | 2100    | USD
"""

MEDIUM_HARD_DATASET = """\
row_id | user_id | amount    | transaction_date | status
-------|---------|-----------|------------------|----------
0      | U001    | 150.00    | 2024-01-10       | completed
1      | U001    | 150.00    | 2024-01-10       | completed
2      | U002    | 99999.99  | 2024-01-11       | completed
3      | U003    | -50.00    | 2024-01-12       | completed
4      | U004    | 200.00    | 2024-01-13       | completed
5      | U005    | 175.00    | 2024-01-14       | COMPLETED
"""

HARD_DATASET = """\
row_id | order_id | customer_id | product_id | qty  | unit_price | order_dt
-------|----------|-------------|------------|------|------------|------------
0      | ORD-001  | C100        | P200       | 2    | 49.99      | 2024-01-05
1      | ORD-002  | C101        | P201       | -1   | 75.00      | 2024-01-06
2      | ORD-003  | C999        | P202       | 3    | 0.00       | 2024-01-07
3      | ORD-004  | C102        | P999       | 1    | 29.99      | 2024-13-01
4      | ORD-005  | C103        | P203       | 1000 | 5.00       | 2024-01-09
5      | ORD-001  | C100        | P200       | 2    | 49.99      | 2024-01-05
"""

ADVERSARIAL_DATASET = """\
row_id | sensor_id | reading   | unit | recorded_at
-------|-----------|-----------|------|------------------
0      | S001      | -17.5     | C    | 2024-01-15 08:00
1      | S002      | 212       | F    | 2024-01-15 08:01
2      | S003      | NULL      | C    | 2024-01-15 08:02
3      | S004      | 98.6      | F    | 2024-01-15 08:03
4      | S005      | 0.0       | C    | 2024-01-15 08:04
5      | S006      | 999       | C    | 2024-01-15 08:05
"""

TITANIC_FALLBACK_DATASET = """\
row_id | PassengerId | Name                                         | Sex    | Age  | Fare    | Cabin | Embarked | Ticket   | Pclass
-------|-------------|----------------------------------------------|--------|------|---------|-------|----------|----------|-------
0      | 6           | Moran, Mr. James                             | male   | NULL | 8.4583  | NULL  | Q        | 330877   | 3
1      | 62          | Icard, Miss. Amelie                          | female | 38.0 | 80.0000 | B28   | NULL     | 113572   | 1
2      | 830         | Stone, Mrs. George Nelson (Martha Evelyn)    | female | 62.0 | 80.0000 | B28   | NULL     | 113572   | 1
3      | 1           | Braund, Mr. Owen Harris                      | male   | 22.0 | 7.2500  | NULL  | S        | A/5 21171| 3
4      | 2           | Cumings, Mrs. John Bradley (Florence Briggs) | female | 38.0 | 71.2833 | C85   | C        | PC 17599 | 1
"""


def _normalize_cell(value: str | None) -> str:
    if value is None:
        return "NULL"
    cleaned = str(value).strip()
    return cleaned if cleaned else "NULL"


def _titanic_task_preview() -> str:
    dataset_path = Path("data/kaggle/Titanic-Dataset.csv")
    if not dataset_path.exists():
        return TITANIC_FALLBACK_DATASET

    target_ids = [6, 62, 830, 1, 2]
    selected_rows: list[dict[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            passenger_id = int(row["PassengerId"])
            if passenger_id in target_ids:
                selected_rows.append(row)
            if len(selected_rows) == len(target_ids):
                break

    if len(selected_rows) != len(target_ids):
        return TITANIC_FALLBACK_DATASET

    ordered_rows = sorted(selected_rows, key=lambda row: target_ids.index(int(row["PassengerId"])))
    lines = [
        "row_id | PassengerId | Name                                         | Sex    | Age  | Fare    | Cabin | Embarked | Ticket    | Pclass",
        "-------|-------------|----------------------------------------------|--------|------|---------|-------|----------|-----------|-------",
    ]
    for row_index, row in enumerate(ordered_rows):
        lines.append(
            f"{row_index:<6} | "
            f"{row['PassengerId']:<11} | "
            f"{row['Name'][:44]:<44} | "
            f"{row['Sex']:<6} | "
            f"{_normalize_cell(row.get('Age')):<4} | "
            f"{_normalize_cell(row.get('Fare')):<7} | "
            f"{_normalize_cell(row.get('Cabin')):<5} | "
            f"{_normalize_cell(row.get('Embarked')):<8} | "
            f"{_normalize_cell(row.get('Ticket'))[:9]:<9} | "
            f"{_normalize_cell(row.get('Pclass')):<5}"
        )
    return "\n".join(lines)


TASKS: dict[str, Task] = {
    "null_filling": Task(
        id="null_filling",
        name="Missing Value Repair",
        description=(
            "A contacts table has missing and malformed null values. "
            "Identify and fill each missing field with a sensible placeholder."
        ),
        difficulty=Difficulty.EASY,
        max_steps=5,
        config={
            "dataset_preview": EASY_DATASET,
            "available_actions": ["fill_missing", "fix_value", "flag_anomaly"],
            "issues": [
                {"row_index": 1, "column": "age", "issue_type": "missing"},
                {"row_index": 2, "column": "age", "issue_type": "string_null"},
                {"row_index": 4, "column": "email", "issue_type": "missing"},
            ],
        },
        tags=["data-cleaning", "nulls", "contacts"],
        success_criteria="Fill all 3 missing or malformed cells with valid placeholder values.",
    ),
    "format_standardization": Task(
        id="format_standardization",
        name="Date and Currency Standardization",
        description=(
            "A revenue table mixes date formats and currency codes. "
            "Standardize all dates to ISO-8601 and all currency codes to uppercase 3-letter values."
        ),
        difficulty=Difficulty.MEDIUM,
        max_steps=8,
        config={
            "dataset_preview": MEDIUM_DATASET,
            "available_actions": ["standardize_format", "fix_value", "flag_anomaly"],
            "issues": [
                {"row_index": 1, "column": "date", "issue_type": "format", "correct": "2024-01-15"},
                {"row_index": 2, "column": "date", "issue_type": "format", "correct": "2024-01-15"},
                {"row_index": 4, "column": "date", "issue_type": "format", "correct": "2024-01-15"},
                {"row_index": 1, "column": "currency", "issue_type": "case", "correct": "USD"},
                {"row_index": 3, "column": "currency", "issue_type": "format", "correct": "USD"},
            ],
        },
        tags=["data-cleaning", "formats", "dates", "currencies"],
        success_criteria="All dates must be ISO-8601 and all currencies must be uppercase 3-letter codes.",
    ),
    "duplicate_outlier": Task(
        id="duplicate_outlier",
        name="Duplicate and Outlier Removal",
        description=(
            "A transactions table contains a duplicate row, an extreme outlier amount, "
            "a negative amount, and a case inconsistency. Identify and fix all four issues."
        ),
        difficulty=Difficulty.MEDIUM,
        max_steps=8,
        config={
            "dataset_preview": MEDIUM_HARD_DATASET,
            "available_actions": ["drop_row", "fix_value", "flag_anomaly", "standardize_format"],
            "issues": [
                {"row_index": 1, "column": "row_id", "issue_type": "duplicate"},
                {"row_index": 2, "column": "amount", "issue_type": "outlier"},
                {"row_index": 3, "column": "amount", "issue_type": "negative"},
                {"row_index": 5, "column": "status", "issue_type": "case", "correct": "completed"},
            ],
        },
        tags=["data-cleaning", "duplicates", "outliers", "transactions"],
        success_criteria="Duplicate dropped, outlier flagged, negative fixed, and status normalized.",
    ),
    "multi_layer_pipeline": Task(
        id="multi_layer_pipeline",
        name="Multi-Layer Order Data Pipeline",
        description=(
            "An orders table has layered issues: a duplicate row, a negative quantity, "
            "two broken foreign key references, an invalid date, a statistical outlier, and a zero price. "
            "Fix all issues without missing the interactions between them."
        ),
        difficulty=Difficulty.HARD,
        max_steps=12,
        config={
            "dataset_preview": HARD_DATASET,
            "available_actions": ["drop_row", "fix_value", "flag_anomaly", "cast_type", "fill_missing"],
            "issues": [
                {"row_index": 1, "column": "qty", "issue_type": "negative"},
                {"row_index": 2, "column": "customer_id", "issue_type": "invalid_fk"},
                {"row_index": 2, "column": "unit_price", "issue_type": "zero_price"},
                {"row_index": 3, "column": "product_id", "issue_type": "invalid_fk"},
                {"row_index": 3, "column": "order_dt", "issue_type": "invalid_date"},
                {"row_index": 4, "column": "qty", "issue_type": "outlier"},
                {"row_index": 5, "column": "row_id", "issue_type": "duplicate"},
            ],
        },
        tags=["data-cleaning", "pipeline", "fk-violations", "orders"],
        success_criteria="All 7 issues correctly identified and acted on.",
    ),
    "adversarial_sensor": Task(
        id="adversarial_sensor",
        name="Adversarial Sensor Data",
        description=(
            "A sensor table contains values that look suspicious but are physically valid. "
            "Only two values are genuinely wrong. Fix only what is truly broken; modifying valid readings is penalized."
        ),
        difficulty=Difficulty.HARD,
        max_steps=8,
        config={
            "dataset_preview": ADVERSARIAL_DATASET,
            "available_actions": ["fix_value", "flag_anomaly", "drop_row", "fill_missing"],
            "issues": [
                {"row_index": 2, "column": "reading", "issue_type": "null"},
                {"row_index": 5, "column": "reading", "issue_type": "outlier"},
            ],
            "traps": [0, 1, 4],
        },
        tags=["data-cleaning", "adversarial", "sensor", "domain-knowledge"],
        success_criteria="Fix the 2 real issues and leave the 3 valid trap rows untouched.",
    ),
    "titanic_manifest": Task(
        id="titanic_manifest",
        name="Titanic Manifest Cleanup",
        description=(
            "A small slice of the Titanic passenger manifest contains real missing values from the Kaggle dataset. "
            "Fill missing Age with a plausible numeric value, fill missing Embarked with a valid port code, "
            "and replace missing Cabin with a consistent placeholder such as Unknown."
        ),
        difficulty=Difficulty.MEDIUM,
        max_steps=6,
        config={
            "dataset_preview": _titanic_task_preview(),
            "available_actions": ["fill_missing", "fix_value", "flag_anomaly"],
            "issues": [
                {
                    "row_index": 0,
                    "column": "Age",
                    "issue_type": "missing",
                    "value_type": "numeric_range",
                    "min": 0,
                    "max": 80,
                },
                {
                    "row_index": 1,
                    "column": "Embarked",
                    "issue_type": "missing",
                    "allowed_values": ["C", "Q", "S"],
                },
                {
                    "row_index": 2,
                    "column": "Embarked",
                    "issue_type": "missing",
                    "allowed_values": ["C", "Q", "S"],
                },
                {
                    "row_index": 3,
                    "column": "Cabin",
                    "issue_type": "missing",
                    "allowed_values": ["Unknown", "UNKNOWN", "missing", "MISSING"],
                },
            ],
        },
        tags=["data-cleaning", "titanic", "missing-values", "demo"],
        success_criteria="Repair the 4 missing manifest fields using plausible values and consistent placeholders.",
    ),
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Task '{task_id}' not found. Available: {available}")
    return TASKS[task_id]


def get_tasks_by_difficulty(difficulty: Difficulty) -> list[Task]:
    return [task for task in TASKS.values() if task.difficulty == difficulty]


def get_default_task() -> Task:
    easy = get_tasks_by_difficulty(Difficulty.EASY)
    return easy[0] if easy else next(iter(TASKS.values()))
