from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_classifier_gate(
    result: dict[str, Any],
    minimum_accuracy: float,
    minimum_balanced_accuracy: float,
    minimum_worst_decile_accuracy: float,
    minimum_top5_accuracy: float,
    maximum_validation_test_gap: float,
) -> dict[str, Any]:
    validation = result["validation"]
    test = result["test"]
    checks = [
        {
            "name": "validation_accuracy",
            "value": validation["accuracy"],
            "threshold": minimum_accuracy,
            "passed": validation["accuracy"] >= minimum_accuracy,
        },
        {
            "name": "test_accuracy",
            "value": test["accuracy"],
            "threshold": minimum_accuracy,
            "passed": test["accuracy"] >= minimum_accuracy,
        },
        {
            "name": "validation_balanced_accuracy",
            "value": validation["balanced_accuracy"],
            "threshold": minimum_balanced_accuracy,
            "passed": validation["balanced_accuracy"] >= minimum_balanced_accuracy,
        },
        {
            "name": "validation_worst_decile_accuracy",
            "value": validation["worst_decile_accuracy"],
            "threshold": minimum_worst_decile_accuracy,
            "passed": validation["worst_decile_accuracy"] >= minimum_worst_decile_accuracy,
        },
        {
            "name": "validation_top5_accuracy",
            "value": validation["top5_accuracy"],
            "threshold": minimum_top5_accuracy,
            "passed": validation["top5_accuracy"] >= minimum_top5_accuracy,
        },
        {
            "name": "validation_test_gap",
            "value": abs(validation["accuracy"] - test["accuracy"]),
            "threshold": maximum_validation_test_gap,
            "passed": abs(validation["accuracy"] - test["accuracy"]) <= maximum_validation_test_gap,
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "best_step": result["best_step"],
        "steps": result["steps"],
        "model_sha256": result["best_model_sha256"],
    }


def require_classifier_gate(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "stage_01_classifier_gate.json"
    if not path.exists():
        raise FileNotFoundError(f"Classifier quality gate is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed", False):
        raise RuntimeError(f"Classifier quality gate failed: {path}")
    return result
