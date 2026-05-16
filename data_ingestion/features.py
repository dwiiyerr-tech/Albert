from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_path(root: Path, pattern: str) -> Optional[Path]:
    files = [p for p in root.glob(pattern) if p.is_file() and p.stat().st_size > 0]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _date_key(value: str) -> str:
    return str(value)[:10]


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [v for v in values if v is not None]
    return sum(items) / len(items) if items else None


def _stdev(values: Iterable[float]) -> Optional[float]:
    items = [v for v in values if v is not None]
    if len(items) < 2:
        return None
    return statistics.stdev(items)


def _daily_funding(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rate = _float(row.get("funding_rate"))
        if rate is None:
            continue
        grouped[_date_key(row.get("funding_time", ""))].append(rate)
    return {day: sum(values) / len(values) for day, values in grouped.items()}


def _single_value_by_date(rows: list[dict], date_field: str, value_field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        value = _float(row.get(value_field))
        if value is not None:
            out[_date_key(row.get(date_field, ""))] = value
    return out


def _regime_label(
    ret_1d: Optional[float],
    ret_7d: Optional[float],
    vol_30d_ann: Optional[float],
    drawdown: Optional[float],
    close: float,
    ma_30: Optional[float],
    ma_200: Optional[float],
) -> str:
    if ret_1d is not None and ret_1d <= -0.10:
        return "crash"
    if ret_7d is not None and ret_7d <= -0.20:
        return "crash"
    if vol_30d_ann is not None and vol_30d_ann >= 0.85:
        return "high_vol"
    if drawdown is not None and drawdown <= -0.35:
        if ma_30 is not None and close >= ma_30:
            return "recovery"
        return "deep_drawdown"
    if drawdown is not None and drawdown <= -0.15 and ma_30 is not None and close < ma_30:
        return "risk_off"
    if ma_30 is not None and ma_200 is not None and close < ma_30 < ma_200:
        return "risk_off"
    if ma_30 is not None and ma_200 is not None and close > ma_30 > ma_200:
        if vol_30d_ann is None or vol_30d_ann < 0.65:
            return "risk_on"
    return "neutral"


def _risk_multiplier(
    regime: str,
    funding_avg: Optional[float],
    long_short_ratio: Optional[float],
) -> float:
    base = {
        "crash": 0.25,
        "deep_drawdown": 0.55,
        "high_vol": 0.50,
        "recovery": 0.80,
        "risk_off": 0.65,
        "neutral": 1.00,
        "risk_on": 1.10,
    }.get(regime, 1.0)

    # Crowded leverage should reduce size even if trend looks healthy.
    if funding_avg is not None and abs(funding_avg) >= 0.0005:
        base *= 0.90
    if long_short_ratio is not None and (long_short_ratio >= 1.8 or long_short_ratio <= 0.55):
        base *= 0.90

    return round(max(0.20, min(1.10, base)), 3)


def build_btc_risk_regimes(
    ohlcv_rows: list[dict],
    funding_rows: list[dict] | None = None,
    open_interest_rows: list[dict] | None = None,
    long_short_rows: list[dict] | None = None,
) -> list[dict]:
    candles = sorted(ohlcv_rows, key=lambda row: row.get("open_time", ""))
    funding = _daily_funding(funding_rows or [])
    open_interest = _single_value_by_date(open_interest_rows or [], "timestamp", "sum_open_interest_value")
    long_short = _single_value_by_date(long_short_rows or [], "timestamp", "long_short_ratio")

    closes: list[float] = []
    log_returns: list[float] = []
    running_high = 0.0
    out: list[dict] = []

    for row in candles:
        close = _float(row.get("close"))
        if close is None or close <= 0:
            continue

        date = _date_key(row.get("open_time", ""))
        prev_close = closes[-1] if closes else None
        ret_1d = (close / prev_close - 1.0) if prev_close else None
        ret_7d = (close / closes[-7] - 1.0) if len(closes) >= 7 else None
        if prev_close:
            log_returns.append(math.log(close / prev_close))

        closes.append(close)
        running_high = max(running_high, close)
        drawdown = close / running_high - 1.0 if running_high else None
        ma_7 = _mean(closes[-7:])
        ma_30 = _mean(closes[-30:]) if len(closes) >= 30 else None
        ma_200 = _mean(closes[-200:]) if len(closes) >= 200 else None
        vol_7d = _stdev(log_returns[-7:]) * math.sqrt(365) if len(log_returns) >= 7 else None
        vol_30d = _stdev(log_returns[-30:]) * math.sqrt(365) if len(log_returns) >= 30 else None

        funding_avg = funding.get(date)
        oi_value = open_interest.get(date)
        ls_ratio = long_short.get(date)
        regime = _regime_label(ret_1d, ret_7d, vol_30d, drawdown, close, ma_30, ma_200)

        out.append({
            "date": date,
            "symbol": row.get("symbol", "BTCUSDT"),
            "close": close,
            "ret_1d": round(ret_1d, 6) if ret_1d is not None else None,
            "ret_7d": round(ret_7d, 6) if ret_7d is not None else None,
            "drawdown_from_ath": round(drawdown, 6) if drawdown is not None else None,
            "ma_7": round(ma_7, 6) if ma_7 is not None else None,
            "ma_30": round(ma_30, 6) if ma_30 is not None else None,
            "ma_200": round(ma_200, 6) if ma_200 is not None else None,
            "realized_vol_7d_ann": round(vol_7d, 6) if vol_7d is not None else None,
            "realized_vol_30d_ann": round(vol_30d, 6) if vol_30d is not None else None,
            "funding_rate_avg": round(funding_avg, 8) if funding_avg is not None else None,
            "open_interest_value": oi_value,
            "long_short_ratio": ls_ratio,
            "regime": regime,
            "risk_multiplier": _risk_multiplier(regime, funding_avg, ls_ratio),
        })

    return out


def build_weather_monthly_normals(actual_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in actual_rows:
        try:
            month = dt.date.fromisoformat(row["date"]).month
        except (KeyError, ValueError):
            continue
        grouped[(str(row.get("city", "")), month)].append(row)

    out: list[dict] = []
    for (city, month), rows in sorted(grouped.items()):
        max_values = [_float(r.get("temperature_2m_max_c")) for r in rows]
        min_values = [_float(r.get("temperature_2m_min_c")) for r in rows]
        precip_values = [_float(r.get("precipitation_sum_mm")) for r in rows]
        out.append({
            "city": city,
            "month": month,
            "sample_days": len(rows),
            "avg_temperature_2m_max_c": round(_mean(max_values) or 0.0, 4),
            "std_temperature_2m_max_c": round(_stdev(max_values) or 0.0, 4),
            "avg_temperature_2m_min_c": round(_mean(min_values) or 0.0, 4),
            "std_temperature_2m_min_c": round(_stdev(min_values) or 0.0, 4),
            "avg_precipitation_sum_mm": round(_mean(precip_values) or 0.0, 4),
        })
    return out
