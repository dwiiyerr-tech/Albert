"""
Experience Memory — persistent store for every prediction, trade, outcome,
market observation, and learned lesson.

This is the "long-term memory" of MiroWeather: everything the agent has ever
seen, predicted, got wrong, or discovered about Polymarket is recorded here
so future cycles can learn from the full history.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

MEMORY_FILE = "memory.json"


# ─── Data records ─────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """One prediction made by the simulation, with eventual outcome."""
    id: str
    city: str
    target_date: str
    bucket_low: float
    bucket_high: float
    # Simulation output
    consensus_probability: float
    confidence_level: str
    model_spread_f: Optional[float]
    agent_probabilities: list[float]          # final-round per-agent estimates
    # Market context at time of prediction
    market_price: float                       # YES contract price
    market_volume: float
    hours_to_resolution: float
    market_id: str
    # Forecast model readings
    ecmwf_f: Optional[float]
    gfs_f: Optional[float]
    metar_f: Optional[float]
    # Outcome (filled in after resolution)
    actual_temp_f: Optional[float] = None
    outcome_yes: Optional[bool] = None        # did the YES resolve True?
    resolved: bool = False
    resolution_ts: Optional[str] = None
    # Trade (if executed)
    trade_direction: Optional[str] = None     # YES / NO / None
    trade_size_usd: Optional[float] = None
    trade_ev: Optional[float] = None
    trade_pnl_usd: Optional[float] = None
    # Metadata
    created_ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    @property
    def prediction_error(self) -> Optional[float]:
        """Signed error: predicted_p - actual (0 or 1)."""
        if self.outcome_yes is None:
            return None
        actual = 1.0 if self.outcome_yes else 0.0
        return self.consensus_probability - actual

    @property
    def brier_score(self) -> Optional[float]:
        """Lower is better. Perfect = 0, worst = 1."""
        if self.outcome_yes is None:
            return None
        actual = 1.0 if self.outcome_yes else 0.0
        return (self.consensus_probability - actual) ** 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketObservation:
    """
    A snapshot of a Polymarket weather market — price, volume, spread,
    timing — logged regardless of whether we traded it.
    Accumulates over time to reveal structural market patterns.
    """
    id: str
    city: str
    market_id: str
    question: str
    bucket_low: float
    bucket_high: float
    price_yes: float
    price_no: float
    spread: float
    volume: float
    hours_to_resolution: float
    observed_ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    # What actually happened (filled after resolution)
    resolved_yes: Optional[bool] = None
    actual_temp_f: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MarketObservation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Lesson:
    """
    A structured insight extracted by the reflection engine.
    Lessons are injected into future agent debates and trading decisions.
    """
    id: str
    category: str           # "model_bias" | "market_pattern" | "city_specific" |
                            # "timing" | "agent_calibration" | "risk" | "general"
    city: Optional[str]     # None = global lesson
    content: str            # natural-language lesson text
    confidence: float       # 0–1 how confident we are in this lesson
    supporting_records: list[str]  # PredictionRecord IDs
    created_ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    times_applied: int = 0
    times_validated: int = 0  # how many times subsequent predictions confirmed it
    times_violated: int = 0   # how many times it was contradicted

    @property
    def reliability_score(self) -> float:
        total = self.times_validated + self.times_violated
        if total == 0:
            return self.confidence
        return self.times_validated / total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Lesson":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Memory store ─────────────────────────────────────────────────────────────

class ExperienceMemory:
    """
    Append-only experience store persisted as JSON.
    Provides retrieval methods for the learning and simulation subsystems.
    """

    def __init__(self, memory_file: str = MEMORY_FILE) -> None:
        self._file = memory_file
        self.predictions: dict[str, PredictionRecord] = {}
        self.observations: list[MarketObservation] = []
        self.lessons: list[Lesson] = []
        self._load()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file) as f:
                raw = json.load(f)
            for d in raw.get("predictions", []):
                rec = PredictionRecord.from_dict(d)
                self.predictions[rec.id] = rec
            for d in raw.get("observations", []):
                self.observations.append(MarketObservation.from_dict(d))
            for d in raw.get("lessons", []):
                self.lessons.append(Lesson.from_dict(d))
            logger.info(
                "Memory loaded: %d predictions, %d observations, %d lessons",
                len(self.predictions), len(self.observations), len(self.lessons),
            )
        except Exception as exc:
            logger.warning("Memory load failed: %s", exc)

    def save(self) -> None:
        payload = {
            "predictions": [r.to_dict() for r in self.predictions.values()],
            "observations": [o.to_dict() for o in self.observations[-2000:]],
            "lessons": [l.to_dict() for l in self.lessons],
        }
        with open(self._file, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug("Memory saved")

    # ─── Write ────────────────────────────────────────────────────────────────

    def record_prediction(self, **kwargs) -> PredictionRecord:
        rec = PredictionRecord(id=str(uuid.uuid4()), **kwargs)
        self.predictions[rec.id] = rec
        return rec

    def record_observation(self, **kwargs) -> MarketObservation:
        obs = MarketObservation(id=str(uuid.uuid4()), **kwargs)
        self.observations.append(obs)
        return obs

    def add_lesson(self, lesson: Lesson) -> None:
        # Deduplicate by content similarity (simple: exact match)
        existing = [l for l in self.lessons if l.content.strip() == lesson.content.strip()]
        if not existing:
            self.lessons.append(lesson)
            logger.info("New lesson added [%s]: %s", lesson.category, lesson.content[:80])

    def resolve_prediction(
        self,
        record_id: str,
        actual_temp_f: float,
        outcome_yes: bool,
        pnl_usd: Optional[float] = None,
    ) -> Optional[PredictionRecord]:
        rec = self.predictions.get(record_id)
        if not rec:
            return None
        rec.actual_temp_f = actual_temp_f
        rec.outcome_yes = outcome_yes
        rec.resolved = True
        rec.resolution_ts = datetime.datetime.utcnow().isoformat()
        if pnl_usd is not None:
            rec.trade_pnl_usd = pnl_usd
        return rec

    # ─── Read / retrieval ─────────────────────────────────────────────────────

    def resolved_records(self, city: Optional[str] = None, last_n: int = 200) -> list[PredictionRecord]:
        recs = [r for r in self.predictions.values() if r.resolved]
        if city:
            recs = [r for r in recs if r.city == city]
        recs.sort(key=lambda r: r.created_ts, reverse=True)
        return recs[:last_n]

    def unresolved_records(self) -> list[PredictionRecord]:
        return [r for r in self.predictions.values() if not r.resolved]

    def lessons_for_city(self, city: str, top_n: int = 8) -> list[Lesson]:
        """Return most reliable lessons applicable to a given city."""
        relevant = [
            l for l in self.lessons
            if l.city is None or l.city == city
        ]
        relevant.sort(key=lambda l: -l.reliability_score)
        return relevant[:top_n]

    def lessons_by_category(self, category: str) -> list[Lesson]:
        return [l for l in self.lessons if l.category == category]

    def recent_market_stats(self, city: str, last_n: int = 50) -> dict:
        """Aggregate stats from recent resolved observations for a city."""
        city_obs = [
            o for o in self.observations
            if o.city == city and o.resolved_yes is not None
        ][-last_n:]
        if not city_obs:
            return {}
        avg_spread = sum(o.spread for o in city_obs) / len(city_obs)
        avg_volume = sum(o.volume for o in city_obs) / len(city_obs)
        yes_rate = sum(1 for o in city_obs if o.resolved_yes) / len(city_obs)
        return {
            "n": len(city_obs),
            "avg_spread": avg_spread,
            "avg_volume": avg_volume,
            "yes_resolution_rate": yes_rate,
        }

    def overall_stats(self) -> dict:
        resolved = self.resolved_records()
        if not resolved:
            return {"total_predictions": 0}
        briers = [r.brier_score for r in resolved if r.brier_score is not None]
        avg_brier = sum(briers) / len(briers) if briers else None
        profitable = [r for r in resolved if r.trade_pnl_usd is not None and r.trade_pnl_usd > 0]
        losing = [r for r in resolved if r.trade_pnl_usd is not None and r.trade_pnl_usd <= 0]
        total_pnl = sum(r.trade_pnl_usd for r in resolved if r.trade_pnl_usd is not None)
        return {
            "total_predictions": len(resolved),
            "avg_brier_score": round(avg_brier, 4) if avg_brier else None,
            "total_trades": len(profitable) + len(losing),
            "win_rate": len(profitable) / (len(profitable) + len(losing)) if (profitable or losing) else None,
            "total_pnl_usd": round(total_pnl, 2),
            "total_lessons": len(self.lessons),
        }
