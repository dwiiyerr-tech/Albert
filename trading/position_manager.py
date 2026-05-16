"""
Position Manager — from WeatherBot (alteregoeth-ai/weatherbot)

Manages open simulated positions with stop-loss and trailing stop logic.
All trades are paper-traded (simulation only) unless LIVE_TRADING=true.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field, fields
from typing import Optional

from config import STOP_LOSS_PCT, TRAILING_STOP_TRIGGER

logger = logging.getLogger(__name__)

POSITIONS_FILE = "positions.json"


@dataclass
class Position:
    market_id: str
    city: str
    direction: str          # YES or NO
    entry_price: float
    current_price: float
    size_usd: float
    opened_at: str
    bucket_low: float
    bucket_high: float
    target_date: str
    order_id: str = ""       # CLOB order ID from Polymarket; empty for paper/demo trades
    no_token_id: str = ""     # CLOB NO token ID; needed to mark NO positions to market
    stop_price: float = 0.0
    trailing_active: bool = False
    closed: bool = False
    close_reason: str = ""
    closed_at: str = ""
    pnl_usd: float = 0.0

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def unrealized_pnl_usd(self) -> float:
        return self.size_usd * self.unrealized_pnl_pct

    @property
    def planned_risk_usd(self) -> float:
        if self.entry_price <= 0:
            return self.size_usd
        risk_fraction = max(0.0, (self.entry_price - self.stop_price) / self.entry_price)
        return self.size_usd * risk_fraction

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class PositionManager:
    """
    Tracks open positions, enforces stop-loss / trailing-stop rules,
    and records closed positions.
    """

    def __init__(self, positions_file: str = POSITIONS_FILE) -> None:
        self._file = positions_file
        self.open_positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file) as f:
                raw = json.load(f)
            for pid, d in raw.get("open", {}).items():
                self.open_positions[pid] = Position.from_dict(d)
            for d in raw.get("closed", []):
                self.closed_positions.append(Position.from_dict(d))
            logger.info("Loaded %d open, %d closed positions",
                        len(self.open_positions), len(self.closed_positions))
        except Exception as exc:
            logger.warning("Could not load positions: %s", exc)

    def save(self) -> None:
        payload = {
            "open": {pid: p.to_dict() for pid, p in self.open_positions.items()},
            "closed": [p.to_dict() for p in self.closed_positions[-200:]],
        }
        with open(self._file, "w") as f:
            json.dump(payload, f, indent=2)

    # ─── Position lifecycle ───────────────────────────────────────────────────

    def open_position(
        self,
        market_id: str,
        city: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        bucket_low: float,
        bucket_high: float,
        target_date: str,
        order_id: str = "",
        no_token_id: str = "",
    ) -> Position:
        # The executor buys the selected outcome token. YES and NO positions are
        # both long token exposures, so loss control is below the entry price.
        direction = direction.upper()
        stop = entry_price * (1 - STOP_LOSS_PCT)

        pos = Position(
            market_id=market_id,
            city=city,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            size_usd=size_usd,
            opened_at=datetime.datetime.utcnow().isoformat(),
            bucket_low=bucket_low,
            bucket_high=bucket_high,
            target_date=target_date,
            order_id=order_id,
            no_token_id=no_token_id,
            stop_price=stop,
        )
        self.open_positions[market_id] = pos
        logger.info("Opened %s %s @ %.3f, stop @ %.3f", direction, city, entry_price, stop)
        return pos

    def update_price(
        self,
        market_id: str,
        current_price: float,
        *,
        close_on_trigger: bool = True,
    ) -> Optional[str]:
        """Update price and check exit conditions. Returns close_reason or None."""
        pos = self.open_positions.get(market_id)
        if not pos:
            return None
        pos.current_price = current_price

        # Positions track the price of the token we bought. Profit always means
        # the owned token price rose, whether that token is YES or NO.
        profit_pct = (current_price - pos.entry_price) / pos.entry_price

        if not pos.trailing_active and profit_pct >= TRAILING_STOP_TRIGGER:
            pos.trailing_active = True
            pos.stop_price = pos.entry_price  # move stop to breakeven
            logger.info("Trailing stop activated for %s %s", pos.city, market_id)

        # Update trailing stop in the favourable direction
        if pos.trailing_active:
            new_stop = current_price * (1 - STOP_LOSS_PCT)
            if new_stop > pos.stop_price:
                pos.stop_price = new_stop

        # Check stop-loss trigger
        if current_price <= pos.stop_price:
            if close_on_trigger:
                self._close(pos, "stop_loss")
            return "stop_loss"

        return None

    def close_position(self, market_id: str, reason: str = "manual") -> Optional[Position]:
        pos = self.open_positions.get(market_id)
        if not pos:
            return None
        return self._close(pos, reason)

    def resolve_position(
        self,
        market_id: str,
        outcome_yes: bool,
        reason: str = "resolved",
    ) -> Optional[Position]:
        """
        Close a binary outcome position at final payout.

        YES token pays 1.0 when outcome_yes is True; NO token pays 1.0 when
        outcome_yes is False. The position key remains the YES token/market id
        so opposite-side duplicate exposure is still blocked.
        """
        yes_payout = 1.0 if outcome_yes else 0.0
        no_payout = 0.0 if outcome_yes else 1.0
        return self.resolve_position_payout(market_id, yes_payout, no_payout, reason)

    def resolve_position_payout(
        self,
        market_id: str,
        yes_payout: float,
        no_payout: float,
        reason: str = "resolved",
    ) -> Optional[Position]:
        """Close a position using explicit YES/NO settlement payouts."""
        pos = self.open_positions.get(market_id)
        if not pos:
            return None
        payout = yes_payout if pos.direction == "YES" else no_payout
        pos.current_price = max(0.0, min(1.0, payout))
        return self._close(pos, reason)

    def _close(self, pos: Position, reason: str) -> Position:
        pos.closed = True
        pos.close_reason = reason
        pos.closed_at = datetime.datetime.utcnow().isoformat()
        pnl_per_dollar = (pos.current_price - pos.entry_price) / pos.entry_price
        pos.pnl_usd = pos.size_usd * pnl_per_dollar
        del self.open_positions[pos.market_id]
        self.closed_positions.append(pos)
        logger.info(
            "Closed %s %s: reason=%s, PnL=%.2f",
            pos.city, pos.market_id, reason, pos.pnl_usd,
        )
        return pos

    # ─── Statistics ───────────────────────────────────────────────────────────

    def total_pnl(self) -> float:
        return sum(p.pnl_usd for p in self.closed_positions)

    def open_deployed_usd(self) -> float:
        return sum(p.size_usd for p in self.open_positions.values())

    def open_planned_risk_usd(self) -> float:
        return sum(p.planned_risk_usd for p in self.open_positions.values())

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl_usd for p in self.open_positions.values())

    def target_date_exposure_usd(self, target_date: str) -> float:
        return sum(
            p.size_usd for p in self.open_positions.values()
            if p.target_date == target_date
        )

    def realized_pnl_since(self, since: datetime.datetime) -> float:
        total = 0.0
        for pos in self.closed_positions:
            if not pos.closed_at:
                continue
            try:
                closed_at = datetime.datetime.fromisoformat(pos.closed_at)
            except ValueError:
                continue
            if closed_at >= since:
                total += pos.pnl_usd
        return total

    def closed_count_since(self, since: datetime.datetime) -> int:
        count = 0
        for pos in self.closed_positions:
            if not pos.closed_at:
                continue
            try:
                closed_at = datetime.datetime.fromisoformat(pos.closed_at)
            except ValueError:
                continue
            if closed_at >= since:
                count += 1
        return count

    def opened_count_since(self, since: datetime.datetime) -> int:
        count = 0
        for pos in list(self.open_positions.values()) + self.closed_positions:
            if not pos.opened_at:
                continue
            try:
                opened_at = datetime.datetime.fromisoformat(pos.opened_at)
            except ValueError:
                continue
            if opened_at >= since:
                count += 1
        return count

    def risk_snapshot(self, now: datetime.datetime | None = None) -> dict:
        now = now or datetime.datetime.utcnow()
        day_start = datetime.datetime.combine(now.date(), datetime.time.min)
        realized_today = self.realized_pnl_since(day_start)
        unrealized = self.total_unrealized_pnl()
        net_pnl = self.total_pnl() + unrealized
        return {
            "open_positions": len(self.open_positions),
            "open_deployed_usd": round(self.open_deployed_usd(), 2),
            "open_planned_risk_usd": round(self.open_planned_risk_usd(), 2),
            "unrealized_pnl_usd": round(unrealized, 2),
            "realized_today_usd": round(realized_today, 2),
            "daily_loss_usd": round(max(0.0, -realized_today - min(0.0, unrealized)), 2),
            "net_pnl_usd": round(net_pnl, 2),
            "drawdown_usd": round(max(0.0, -net_pnl), 2),
            "closed_today": self.closed_count_since(day_start),
            "opened_today": self.opened_count_since(day_start),
        }

    def win_rate(self) -> Optional[float]:
        finished = [p for p in self.closed_positions if p.pnl_usd != 0]
        if not finished:
            return None
        wins = sum(1 for p in finished if p.pnl_usd > 0)
        return wins / len(finished)

    def summary(self) -> dict:
        return {
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "total_pnl_usd": round(self.total_pnl(), 2),
            "unrealized_pnl_usd": round(self.total_unrealized_pnl(), 2),
            "open_deployed_usd": round(self.open_deployed_usd(), 2),
            "open_planned_risk_usd": round(self.open_planned_risk_usd(), 2),
            "win_rate": self.win_rate(),
        }
