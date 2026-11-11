#!/usr/bin/env python3
"""Write CBOT and FCPO daily bars, independent of training.

Sina's global futures daily is pulled here. Volume, open interest, and
settlement are not on this feed. Only the futures daemon should call update(), at 14:45
and 20:25, so a day without training still gets a file.
Other jobs only read the files already on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import train

LABELS = {"S": "美豆", "SM": "美豆粕", "BO": "美豆油", "FCPO": "马棕油"}


def update() -> dict[str, dict]:
    """Fetch and overwrite every overseas daily file. Returns the last row of each."""

    out: dict[str, dict] = {}
    errors: list[str] = []
    for key, symbol in {**train.OVERSEAS, **train.OVERSEAS_EXTRA}.items():
        path = train.series_path(key)
        try:
            rows = train.fetch_overseas(symbol)
            if not rows:
                raise RuntimeError("空行情")
        except Exception as exc:
            errors.append(f"{LABELS.get(key, key)} {exc}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        out[key] = rows[-1]
    if errors:
        raise RuntimeError("；".join(errors))
    return out


if __name__ == "__main__":
    for key, row in update().items():
        print(f"{LABELS.get(key, key)} {row['日期']} 收盘 {row['收盘']}", flush=True)
