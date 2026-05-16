from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RiskRegime:
    date: str
    regime: str
    risk_multiplier: float
    realized_vol_30d_ann: Optional[float] = None
    drawdown_from_ath: Optional[float] = None


class RiskRegimeStore:
    """Loads processed BTC risk-regime features for conservative sizing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._latest: Optional[RiskRegime] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        latest_row: Optional[dict] = None
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    latest_row = json.loads(line)
        if not latest_row:
            return
        self._latest = RiskRegime(
            date=str(latest_row.get("date") or ""),
            regime=str(latest_row.get("regime") or "neutral"),
            risk_multiplier=float(latest_row.get("risk_multiplier") or 1.0),
            realized_vol_30d_ann=latest_row.get("realized_vol_30d_ann"),
            drawdown_from_ath=latest_row.get("drawdown_from_ath"),
        )

    def latest(self) -> Optional[RiskRegime]:
        return self._latest
