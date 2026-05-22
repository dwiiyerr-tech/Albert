from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .memory import ExperienceMemory, PredictionRecord


@dataclass
class DecisionScorecard:
    resolved_predictions: int = 0
    traded_predictions: int = 0
    avg_brier_score: float | None = None
    calibration_bias: float | None = None
    total_pnl_usd: float = 0.0
    win_rate: float | None = None
    avg_trade_ev: float | None = None
    profit_factor: float | None = None
    by_confidence: dict[str, dict] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "resolved_predictions": self.resolved_predictions,
            "traded_predictions": self.traded_predictions,
            "avg_brier_score": self.avg_brier_score,
            "calibration_bias": self.calibration_bias,
            "total_pnl_usd": self.total_pnl_usd,
            "win_rate": self.win_rate,
            "avg_trade_ev": self.avg_trade_ev,
            "profit_factor": self.profit_factor,
            "by_confidence": self.by_confidence,
            "recommendations": list(self.recommendations),
        }


def _direction_won(record: PredictionRecord) -> bool | None:
    if record.outcome_yes is None or not record.trade_direction:
        return None
    direction = record.trade_direction.upper()
    if direction == "YES":
        return bool(record.outcome_yes)
    if direction == "NO":
        return not bool(record.outcome_yes)
    return None


def _simulated_trade_pnl(record: PredictionRecord) -> float | None:
    if record.trade_pnl_usd is not None:
        return float(record.trade_pnl_usd)
    if record.trade_direction is None or record.trade_size_usd is None:
        return None
    won = _direction_won(record)
    if won is None:
        return None
    yes_price = float(record.market_price or 0.0)
    entry_price = yes_price if record.trade_direction.upper() == "YES" else 1.0 - yes_price
    if entry_price <= 0:
        return None
    payout = 1.0 if won else 0.0
    shares = float(record.trade_size_usd) / entry_price
    return shares * payout - float(record.trade_size_usd)


def build_decision_scorecard(
    memory: ExperienceMemory,
    *,
    last_n: int = 500,
) -> DecisionScorecard:
    records = [
        record for record in memory.resolved_records(last_n=last_n)
        if record.outcome_yes is not None
    ]
    scorecard = DecisionScorecard(resolved_predictions=len(records))
    if not records:
        scorecard.recommendations.append("collect resolved outcomes before tuning thresholds")
        return scorecard

    briers = [record.brier_score for record in records if record.brier_score is not None]
    errors = [record.prediction_error for record in records if record.prediction_error is not None]
    if briers:
        scorecard.avg_brier_score = round(sum(briers) / len(briers), 4)
    if errors:
        scorecard.calibration_bias = round(sum(errors) / len(errors), 4)

    trades = [record for record in records if record.trade_direction]
    scorecard.traded_predictions = len(trades)
    pnls = [_simulated_trade_pnl(record) for record in trades]
    pnls = [pnl for pnl in pnls if pnl is not None]
    if pnls:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        wins = sum(1 for pnl in pnls if pnl > 0)
        scorecard.total_pnl_usd = round(sum(pnls), 2)
        scorecard.win_rate = round(wins / len(pnls), 4)
        scorecard.profit_factor = round(gross_profit / gross_loss, 4) if gross_loss else None

    evs = [float(record.trade_ev) for record in trades if record.trade_ev is not None]
    if evs:
        scorecard.avg_trade_ev = round(sum(evs) / len(evs), 4)

    scorecard.by_confidence = _confidence_breakdown(records)
    scorecard.recommendations = _recommendations(scorecard)
    return scorecard


def _confidence_breakdown(records: Iterable[PredictionRecord]) -> dict[str, dict]:
    groups: dict[str, list[PredictionRecord]] = {}
    for record in records:
        groups.setdefault(record.confidence_level or "unknown", []).append(record)

    rows: dict[str, dict] = {}
    for confidence, items in sorted(groups.items()):
        briers = [record.brier_score for record in items if record.brier_score is not None]
        traded = [record for record in items if record.trade_direction]
        pnls = [_simulated_trade_pnl(record) for record in traded]
        pnls = [pnl for pnl in pnls if pnl is not None]
        rows[confidence] = {
            "resolved": len(items),
            "traded": len(traded),
            "avg_brier": round(sum(briers) / len(briers), 4) if briers else None,
            "total_pnl_usd": round(sum(pnls), 2) if pnls else 0.0,
            "win_rate": round(sum(1 for pnl in pnls if pnl > 0) / len(pnls), 4) if pnls else None,
        }
    return rows


def _recommendations(scorecard: DecisionScorecard) -> list[str]:
    recs: list[str] = []
    if scorecard.resolved_predictions < 30:
        recs.append("keep live sizing conservative until at least 30 resolved predictions exist")
    if scorecard.traded_predictions < 10:
        recs.append("run demo/paper mode longer before loosening decision thresholds")
    if scorecard.avg_brier_score is not None and scorecard.avg_brier_score > 0.25:
        recs.append("probability calibration is weak; raise MIN_PROB_EDGE or lower size")
    if scorecard.calibration_bias is not None:
        if scorecard.calibration_bias > 0.08:
            recs.append("model is overconfident on YES; reduce YES probabilities or require stronger edge")
        elif scorecard.calibration_bias < -0.08:
            recs.append("model is underconfident on YES; review NO-side sizing and market-bias lessons")
    if scorecard.win_rate is not None and scorecard.win_rate < 0.45:
        recs.append("trade win rate is weak; tighten EV and feature-quality gates")
    if scorecard.total_pnl_usd < 0:
        recs.append("recent PnL is negative; keep risk multipliers throttled")
    if not recs:
        recs.append("scorecard is stable; continue collecting data before increasing live risk")
    return recs
