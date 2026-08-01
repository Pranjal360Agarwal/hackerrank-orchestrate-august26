#!/usr/bin/env python3
"""Evaluate router behavior on supplied examples and validate submission CSVs.

Usage (from repo root):
    python code/evaluation/main.py
    python code/evaluation/main.py --output output.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from main import OUTPUT_COLUMNS, Router, read_csv  # noqa: E402


def validate_output(output_path: Path, message_path: Path) -> list[str]:
    """Return human-readable contract errors without changing the prediction file."""
    errors: list[str] = []
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_COLUMNS:
            errors.append(f"columns must be exactly {OUTPUT_COLUMNS}; got {reader.fieldnames}")
        rows = list(reader)
    expected = [row["message_id"] for row in read_csv(message_path)]
    actual = [row.get("message_id", "") for row in rows]
    if len(rows) != len(expected):
        errors.append(f"expected {len(expected)} prediction rows, found {len(rows)}")
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        errors.append(f"message IDs/order differ (missing={missing[:5]}, unexpected={unexpected[:5]})")
    for index, row in enumerate(rows, start=2):
        if row.get("action") not in {"notify", "digest", "mute"}:
            errors.append(f"row {index}: invalid action {row.get('action')!r}")
        if row.get("message_type") not in {
            "personal", "urgent", "event", "payment", "business_update", "promotion",
            "greeting", "forward", "spam", "scam", "unknown",
        }:
            errors.append(f"row {index}: invalid message_type {row.get('message_type')!r}")
        try:
            confidence = float(row.get("confidence", ""))
            if not 0 <= confidence <= 1:
                raise ValueError
        except ValueError:
            errors.append(f"row {index}: confidence must be between 0 and 1")
        if not row.get("reason", "").strip():
            errors.append(f"row {index}: reason is empty")
        if not row.get("evidence_message_ids", "").strip():
            errors.append(f"row {index}: evidence_message_ids is empty")
    return errors


def evaluate_examples(dataset: Path) -> None:
    router = Router(dataset)
    examples = read_csv(dataset / "sample_messages.csv")
    predictions = [router.route(router.media.enrich(row)) for row in examples]
    action_correct = sum(p["action"] == truth["action"] for p, truth in zip(predictions, examples))
    type_correct = sum(p["message_type"] == truth["message_type"] for p, truth in zip(predictions, examples))
    joint_correct = sum(
        p["action"] == truth["action"] and p["message_type"] == truth["message_type"]
        for p, truth in zip(predictions, examples)
    )
    count = len(examples)
    print(f"Sample action accuracy: {action_correct}/{count} ({action_correct / count:.1%})")
    print(f"Sample type accuracy:   {type_correct}/{count} ({type_correct / count:.1%})")
    print(f"Sample joint accuracy:  {joint_correct}/{count} ({joint_correct / count:.1%})")
    failures = [
        (truth["message_id"], truth["action"], truth["message_type"], p["action"], p["message_type"])
        for p, truth in zip(predictions, examples)
        if (p["action"], p["message_type"]) != (truth["action"], truth["message_type"])
    ]
    if failures:
        print("Mismatches (id expected_action/type -> predicted_action/type):")
        for failure in failures:
            print("  %s %s/%s -> %s/%s" % failure)
    print("Prediction distribution:", dict(sorted(Counter(p["action"] for p in predictions).items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output.csv")
    args = parser.parse_args()
    evaluate_examples(args.dataset)
    errors = validate_output(args.output, args.dataset / "messages.csv")
    if errors:
        print("\nOutput validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nOutput validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
