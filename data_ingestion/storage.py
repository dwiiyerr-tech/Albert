from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Iterable


def parse_date(value: str) -> _dt.date:
    return _dt.date.fromisoformat(value)


def utc_ms(day: _dt.date, end_of_day: bool = False) -> int:
    if end_of_day:
        dt = _dt.datetime.combine(day, _dt.time(23, 59, 59, 999000), tzinfo=_dt.timezone.utc)
    else:
        dt = _dt.datetime.combine(day, _dt.time.min, tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms: int | str | None) -> str:
    if ms in (None, ""):
        return ""
    try:
        value = int(ms)
    except (TypeError, ValueError):
        return ""
    return _dt.datetime.fromtimestamp(value / 1000, tz=_dt.timezone.utc).isoformat()


def now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
