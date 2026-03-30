"""Task registry for the data-cleaning environment."""

from __future__ import annotations

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
