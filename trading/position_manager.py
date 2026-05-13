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
    stop_price: float = 0.0
    trailing_active: bool = False
    closed: bool = False
    close_reason: str = ""
    pnl_usd: float = 0.0

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

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
    ) -> Position:
        # Fix: stop-loss direction depends on YES vs NO position.
        # YES (long): stop below entry;  NO (short): stop above entry.
        if direction == "YES":
            stop = entry_price * (1 - STOP_LOSS_PCT)
        else:
            stop = entry_price * (1 + STOP_LOSS_PCT)

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
            stop_price=stop,
        )
        self.open_positions[market_id] = pos
        logger.info("Opened %s %s @ %.3f, stop @ %.3f", direction, city, entry_price, stop)
        return pos

    def update_price(self, market_id: str, current_price: float) -> Optional[str]:
        """Update price and check exit conditions. Returns close_reason or None."""
        pos = self.open_positions.get(market_id)
        if not pos:
            return None
        pos.current_price = current_price

        # Trailing stop activation: profit direction depends on YES vs NO
        if pos.direction == "YES":
            profit_pct = (current_price - pos.entry_price) / pos.entry_price
        else:  # NO: profit when price falls
            profit_pct = (pos.entry_price - current_price) / pos.entry_price

        if not pos.trailing_active and profit_pct >= TRAILING_STOP_TRIGGER:
            pos.trailing_active = True
            pos.stop_price = pos.entry_price  # move stop to breakeven
            logger.info("Trailing stop activated for %s %s", pos.city, market_id)

        # Update trailing stop in the favourable direction
        if pos.trailing_active:
            if pos.direction == "YES":
                new_stop = current_price * (1 - STOP_LOSS_PCT)
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop
            else:
                new_stop = current_price * (1 + STOP_LOSS_PCT)
                if new_stop < pos.stop_price:
                    pos.stop_price = new_stop

        # Check stop-loss trigger
        triggered = (
            (pos.direction == "YES" and current_price <= pos.stop_price)
            or (pos.direction == "NO" and current_price >= pos.stop_price)
        )
        if triggered:
            return self._close(pos, "stop_loss")

        return None

    def close_position(self, market_id: str, reason: str = "manual") -> Optional[Position]:
        pos = self.open_positions.get(market_id)
        if not pos:
            return None
        return self._close(pos, reason)

    def _close(self, pos: Position, reason: str) -> Position:
        pos.closed = True
        pos.close_reason = reason
        # Fix: P&L sign depends on direction.
        # YES profits when price rises; NO profits when price falls.
        if pos.direction == "YES":
            pnl_per_dollar = (pos.current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_per_dollar = (pos.entry_price - pos.current_price) / pos.entry_price
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
            "win_rate": self.win_rate(),
        }
