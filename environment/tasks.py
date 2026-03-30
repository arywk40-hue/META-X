"""Task registry for the code review domain."""

from __future__ import annotations

from .models import Difficulty, Task


EASY_SNIPPET = "\n".join(
    [
        "1  def calculate_average(numbers):",
        "2      \"\"\"Return the arithmetic mean for a list of numbers.\"\"\"",
        "3      total = 0",
        "4      for value in numbers:",
        "5          total += value",
        "6      average = total / len(numbers)",
        "7      return round(average, 2)",
        "8  ",
        "9  sample_data = [10, 20, 30]",
        "10 if __name__ == \"__main__\":",
        "11     print(calculate_average(sample_data))",
        "12 ",
    ]
)

MEDIUM_SNIPPET = "\n".join(
    [
        "1  def binary_search(values, target):",
        "2      \"\"\"Return the index of target in a sorted list or -1.\"\"\"",
        "3      left = 0",
        "4      right = len(values) - 1",
        "5  ",
        "6      if not values:",
        "7          return -1",
        "8  ",
        "9      while left < right:",
        "10         mid = (left + right) // 2",
        "11         guess = values[mid]",
        "12 ",
        "13         if guess == target:",
        "14             return mid",
        "15 ",
        "16         if guess < target:",
        "17             left = mid + 1",
        "18         else:",
        "19             right = mid - 1",
        "20 ",
        "21     if values[left] == target:",
        "22         return left",
        "23 ",
        "24     return -1",
        "25 ",
        "26 sample = [1, 4, 8, 12, 16, 20]",
        "27 print(binary_search(sample, 20))",
        "28 ",
    ]
)

HARD_SNIPPET = "\n".join(
    [
        "1  from fastapi import APIRouter, Depends, HTTPException",
        "2  from pydantic import BaseModel",
        "3  import sqlite3",
        "4  ",
        "5  router = APIRouter(prefix=\"/admin\", tags=[\"admin\"])",
        "6  ",
        "7  class AuditQuery(BaseModel):",
        "8      username: str",
        "9  ",
        "10 def get_db():",
        "11     connection = sqlite3.connect(\"app.db\")",
        "12     connection.row_factory = sqlite3.Row",
        "13     return connection",
        "14 ",
        "15 @router.post(\"/audit-log\")",
        "16 def fetch_audit_log(payload: AuditQuery, db=Depends(get_db)):",
        "17     if not payload.username:",
        "18         raise HTTPException(status_code=400, detail=\"username is required\")",
        "19 ",
        "20     query = (",
        "21         \"SELECT action, created_at \"",
        "22         \"FROM audit_log \"",
        "23         f\"WHERE username = '{payload.username}' \"",
        "24         \"ORDER BY created_at DESC LIMIT 50\"",
        "25     )",
        "26 ",
        "27     cursor = db.execute(query)",
        "28     rows = cursor.fetchall()",
        "29 ",
        "30     return {",
        "31         \"username\": payload.username,",
        "32         \"events\": [dict(row) for row in rows],",
        "33         \"count\": len(rows),",
        "34     }",
        "35 ",
    ]
)


TASKS: dict[str, Task] = {
    "runtime_bug": Task(
        id="runtime_bug",
        name="Runtime Bug Review",
        description=(
            "Review a short Python helper and identify the single runtime bug that would "
            "surface in production edge cases."
        ),
        difficulty=Difficulty.EASY,
        max_steps=3,
        config={
            "code_snippet": EASY_SNIPPET,
            "available_actions": ["report_bug"],
            "answer": {
                "bug_line": 6,
                "bug_type": "runtime",
                "key_terms": ["empty", "division", "zero", "len(numbers)"],
            },
        },
        tags=["code-review", "python", "runtime"],
        success_criteria="Identify the failing line, classify the bug, and explain why it breaks.",
    ),
    "binary_search_logic": Task(
        id="binary_search_logic",
        name="Binary Search Logic Review",
        description=(
            "Review a production-style binary search helper and identify the single logic bug "
            "that breaks edge-case correctness."
        ),
        difficulty=Difficulty.MEDIUM,
        max_steps=3,
        config={
            "code_snippet": MEDIUM_SNIPPET,
            "available_actions": ["report_bug"],
            "answer": {
                "bug_line": 9,
                "bug_type": "logic",
                "key_terms": ["off-by-one", "loop condition", "<=", "last element"],
            },
        },
        tags=["code-review", "python", "logic", "algorithms"],
        success_criteria="Catch the off-by-one error and explain its behavioral impact.",
    ),
    "security_vulnerability": Task(
        id="security_vulnerability",
        name="Security Vulnerability Review",
        description=(
            "Review a realistic FastAPI admin endpoint and identify the single most critical "
            "security bug before it reaches production."
        ),
        difficulty=Difficulty.HARD,
        max_steps=3,
        config={
            "code_snippet": HARD_SNIPPET,
            "available_actions": ["report_bug"],
            "answer": {
                "bug_line": 23,
                "bug_type": "security",
                "key_terms": ["sql injection", "parameterized", "query", "f-string"],
            },
        },
        tags=["code-review", "python", "security", "fastapi"],
        success_criteria="Identify the vulnerable line and explain why the query construction is unsafe.",
    ),
}


def get_task(task_id: str) -> Task:
    """Return a task by ID with a helpful error if it is missing."""
    if task_id not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Task '{task_id}' not found. Available tasks: {available}")
    return TASKS[task_id]


def get_tasks_by_difficulty(difficulty: Difficulty) -> list[Task]:
    """Return all tasks in a given difficulty bucket."""
    return [task for task in TASKS.values() if task.difficulty == difficulty]


def get_default_task() -> Task:
    """Return the first easy task or the first registered task."""
    easy_tasks = get_tasks_by_difficulty(Difficulty.EASY)
    return easy_tasks[0] if easy_tasks else next(iter(TASKS.values()))
