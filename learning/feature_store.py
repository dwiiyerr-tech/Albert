from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from trading.ev_calculator import TradeSignal


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLYMARKET_RAW_DIR = REPO_ROOT / "data" / "raw" / "polymarket"


@dataclass
class MarketFeatures:
    token_id: str = ""
    condition_id: str = ""
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    )
    latest_price: float | None = None
    price_momentum_1h: float | None = None
    price_momentum_6h: float | None = None
    price_momentum_24h: float | None = None
    realized_volatility_24h: float | None = None
    trade_count_1h: int = 0
    trade_count_24h: int = 0
    buy_volume_1h: float = 0.0
    sell_volume_1h: float = 0.0
    trade_imbalance_1h: float | None = None
    holder_count: int = 0
    top_holder_concentration: float | None = None
    open_interest: float | None = None
    open_interest_change_24h: float | None = None
    wallet_activity_count_24h: int = 0
    data_quality_score: float = 0.0
    missing_sources: list[str] = field(default_factory=list)
    stale_sources: list[str] = field(default_factory=list)

    def compact(self) -> dict:
        return {
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "latest_price": self.latest_price,
            "price_momentum_1h": self.price_momentum_1h,
            "price_momentum_6h": self.price_momentum_6h,
            "price_momentum_24h": self.price_momentum_24h,
            "realized_volatility_24h": self.realized_volatility_24h,
            "trade_count_1h": self.trade_count_1h,
            "trade_count_24h": self.trade_count_24h,
            "trade_imbalance_1h": self.trade_imbalance_1h,
            "holder_count": self.holder_count,
            "top_holder_concentration": self.top_holder_concentration,
            "open_interest": self.open_interest,
            "open_interest_change_24h": self.open_interest_change_24h,
            "wallet_activity_count_24h": self.wallet_activity_count_24h,
            "data_quality_score": self.data_quality_score,
            "missing_sources": list(self.missing_sources),
            "stale_sources": list(self.stale_sources),
        }


def _coerce_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _normalise_id(value) -> str:
    return str(value or "").strip().lower()


def _parse_time(value) -> dt.datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds = seconds / 1000
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            seconds = float(text)
            if seconds > 10_000_000_000:
                seconds = seconds / 1000
            return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _row_time(row: dict) -> dt.datetime | None:
    for key in ("timestamp_iso", "collected_at", "created_at", "updated_at", "timestamp", "time", "t"):
        parsed = _parse_time(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


class MarketFeatureStore:
    """Builds decision-ready market features from local Polymarket JSONL data."""

    def __init__(self, raw_dir: str | Path = DEFAULT_POLYMARKET_RAW_DIR) -> None:
        self.raw_dir = Path(raw_dir)
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def features_for_signal(self, signal: "TradeSignal") -> MarketFeatures:
        token_id = signal.no_token_id if signal.direction == "NO" and signal.no_token_id else signal.market_id
        return self.features_for_market(
            token_id=token_id,
            condition_id=getattr(signal, "condition_id", ""),
        )

    def features_for_market(self, token_id: str = "", condition_id: str = "") -> MarketFeatures:
        now = dt.datetime.now(dt.timezone.utc)
        token_key = _normalise_id(token_id)
        condition_key = _normalise_id(condition_id)
        features = MarketFeatures(token_id=token_id, condition_id=condition_id)

        price_rows = [
            row for row in self._load_many("price_history*.jsonl")
            if _normalise_id(row.get("token_id") or row.get("asset") or row.get("market")) == token_key
        ]
        self._apply_price_features(features, price_rows, now)

        trade_rows = [
            row for row in self._load_many("data_api_trades*.jsonl", "clob_trade_events*.jsonl")
            if self._matches_market(row, token_key, condition_key)
        ]
        self._apply_trade_features(features, trade_rows, now)

        holder_rows = [
            row for row in self._load_many("holders*.jsonl")
            if self._matches_market(row, token_key, condition_key)
        ]
        self._apply_holder_features(features, holder_rows)

        oi_rows = [
            row for row in self._load_many("open_interest*.jsonl")
            if self._matches_market(row, token_key, condition_key)
        ]
        self._apply_open_interest_features(features, oi_rows, now)

        activity_rows = [
            row for row in self._load_many("user_activity*.jsonl")
            if self._matches_market(row, token_key, condition_key)
        ]
        features.wallet_activity_count_24h = self._count_recent(activity_rows, now, hours=24)

        self._score_quality(features)
        return features

    def _load_many(self, *patterns: str) -> list[dict]:
        rows: list[dict] = []
        for pattern in patterns:
            for path in sorted(self.raw_dir.glob(pattern)):
                rows.extend(self._load_file(path))
        return rows

    def _load_file(self, path: Path) -> list[dict]:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return []
        key = str(path)
        cached = self._cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        rows = _read_jsonl(path)
        self._cache[key] = (mtime, rows)
        return rows

    def _matches_market(self, row: dict, token_key: str, condition_key: str) -> bool:
        row_token = _normalise_id(row.get("token_id") or row.get("asset") or row.get("market"))
        row_condition = _normalise_id(
            row.get("condition_id") or row.get("conditionId") or row.get("market")
        )
        if token_key and row_token == token_key:
            return True
        if condition_key and row_condition == condition_key:
            return True
        return False

    def _apply_price_features(
        self,
        features: MarketFeatures,
        rows: list[dict],
        now: dt.datetime,
    ) -> None:
        points: list[tuple[dt.datetime, float]] = []
        for row in rows:
            when = _row_time(row)
            price = _coerce_float(row.get("price") if "price" in row else row.get("p"))
            if when is not None and price is not None:
                points.append((when, price))
        points.sort(key=lambda item: item[0])
        if not points:
            features.missing_sources.append("price_history")
            return

        latest_ts, latest_price = points[-1]
        features.latest_price = latest_price
        if (now - latest_ts).total_seconds() > 48 * 3600:
            features.stale_sources.append("price_history")

        features.price_momentum_1h = self._momentum(points, latest_ts, hours=1)
        features.price_momentum_6h = self._momentum(points, latest_ts, hours=6)
        features.price_momentum_24h = self._momentum(points, latest_ts, hours=24)
        features.realized_volatility_24h = self._volatility(points, latest_ts, hours=24)

    def _momentum(
        self,
        points: list[tuple[dt.datetime, float]],
        latest_ts: dt.datetime,
        hours: int,
    ) -> float | None:
        cutoff = latest_ts - dt.timedelta(hours=hours)
        prior = None
        for when, price in points:
            if when <= cutoff:
                prior = price
            else:
                break
        if prior in (None, 0):
            return None
        return (points[-1][1] - prior) / prior

    def _volatility(
        self,
        points: list[tuple[dt.datetime, float]],
        latest_ts: dt.datetime,
        hours: int,
    ) -> float | None:
        cutoff = latest_ts - dt.timedelta(hours=hours)
        recent = [(when, price) for when, price in points if when >= cutoff]
        if len(recent) < 3:
            return None
        returns: list[float] = []
        for idx in range(1, len(recent)):
            prev = recent[idx - 1][1]
            curr = recent[idx][1]
            if prev > 0:
                returns.append((curr - prev) / prev)
        if len(returns) < 2:
            return None
        return statistics.pstdev(returns)

    def _apply_trade_features(
        self,
        features: MarketFeatures,
        rows: list[dict],
        now: dt.datetime,
    ) -> None:
        if not rows:
            features.missing_sources.append("trades")
            return

        recent_1h = self._recent_rows(rows, now, hours=1)
        recent_24h = self._recent_rows(rows, now, hours=24)
        features.trade_count_1h = len(recent_1h)
        features.trade_count_24h = len(recent_24h)
        for row in recent_1h:
            size = _coerce_float(row.get("size")) or 0.0
            price = _coerce_float(row.get("price")) or 1.0
            notional = size * price
            side = str(row.get("side") or "").upper()
            if side == "BUY":
                features.buy_volume_1h += notional
            elif side == "SELL":
                features.sell_volume_1h += notional

        total = features.buy_volume_1h + features.sell_volume_1h
        if total > 0:
            features.trade_imbalance_1h = (features.buy_volume_1h - features.sell_volume_1h) / total
        if not recent_24h:
            features.stale_sources.append("trades")

    def _apply_holder_features(self, features: MarketFeatures, rows: list[dict]) -> None:
        if not rows:
            features.missing_sources.append("holders")
            return
        amounts = [_coerce_float(row.get("amount")) for row in rows]
        amounts = [value for value in amounts if value is not None and value > 0]
        if not amounts:
            features.missing_sources.append("holders")
            return
        wallets = {
            str(row.get("proxy_wallet") or row.get("proxyWallet") or row.get("wallet") or "")
            for row in rows
            if row.get("proxy_wallet") or row.get("proxyWallet") or row.get("wallet")
        }
        features.holder_count = len(wallets) or len(amounts)
        total = sum(amounts)
        features.top_holder_concentration = max(amounts) / total if total > 0 else None

    def _apply_open_interest_features(
        self,
        features: MarketFeatures,
        rows: list[dict],
        now: dt.datetime,
    ) -> None:
        points: list[tuple[dt.datetime, float]] = []
        for row in rows:
            value = _coerce_float(row.get("open_interest") if "open_interest" in row else row.get("value"))
            when = _row_time(row)
            if value is not None:
                points.append((when or now, value))
        points.sort(key=lambda item: item[0])
        if not points:
            features.missing_sources.append("open_interest")
            return

        latest_ts, latest_value = points[-1]
        features.open_interest = latest_value
        if (now - latest_ts).total_seconds() > 48 * 3600:
            features.stale_sources.append("open_interest")

        cutoff = latest_ts - dt.timedelta(hours=24)
        prior = None
        for when, value in points:
            if when <= cutoff:
                prior = value
            else:
                break
        if prior not in (None, 0):
            features.open_interest_change_24h = (latest_value - prior) / prior

    def _recent_rows(self, rows: Iterable[dict], now: dt.datetime, hours: int) -> list[dict]:
        cutoff = now - dt.timedelta(hours=hours)
        recent = []
        for row in rows:
            when = _row_time(row)
            if when is not None and when >= cutoff:
                recent.append(row)
        return recent

    def _count_recent(self, rows: Iterable[dict], now: dt.datetime, hours: int) -> int:
        return len(self._recent_rows(rows, now, hours))

    def _score_quality(self, features: MarketFeatures) -> None:
        score = 0.0
        if features.latest_price is not None and "price_history" not in features.stale_sources:
            score += 0.25
        if features.trade_count_24h > 0 and "trades" not in features.stale_sources:
            score += 0.25
        if features.holder_count > 0:
            score += 0.20
        if features.open_interest is not None and "open_interest" not in features.stale_sources:
            score += 0.20
        if features.wallet_activity_count_24h > 0:
            score += 0.10
        features.data_quality_score = round(min(1.0, score), 4)
