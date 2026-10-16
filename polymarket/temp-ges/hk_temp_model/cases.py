"""Append-only rows for replaying a forecast against the book and the settlement."""

from __future__ import annotations

import json
from pathlib import Path

_CASES_PATH = Path(__file__).resolve().parent / "data" / "cases.jsonl"


def append_case(row: dict, path: Path | None = None) -> None:
    file = path or _CASES_PATH
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
