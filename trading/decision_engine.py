from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    DECISION_CONTRA_FLOW_THRESHOLD,
    DECISION_ENGINE_ENABLED,
    DECISION_LOW_DATA_RISK_MULTIPLIER,
    DECISION_MAX_HOLDER_CONCENTRATION,
    DECISION_MAX_VOLATILITY_24H,
    DECISION_MIN_LIVE_FEATURE_QUALITY,
)
from learning.feature_store import MarketFeatures
from .ev_calculator import TradeSignal


@dataclass
class DecisionResult:
    action: str
    live_ready: bool
    risk_multiplier: float = 1.0
    data_quality_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feature_snapshot: dict = field(default_factory=dict)

    @property
    def should_execute_paper(self) -> bool:
        return self.action in {"PAPER_TRADE", "LIVE_READY"}

    def compact(self) -> dict:
        return {
            "action": self.action,
            "live_ready": self.live_ready,
            "risk_multiplier": self.risk_multiplier,
            "data_quality_score": self.data_quality_score,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


class DecisionEngine:
    """
    Final decision gate between EV and execution.

    It never increases risk. Missing pro-data lowers sizing or live readiness;
    severe microstructure warnings move the signal to WATCH.
    """

    def evaluate(
        self,
        signal: TradeSignal,
        features: MarketFeatures,
        *,
        live_mode: bool = False,
    ) -> DecisionResult:
        if not DECISION_ENGINE_ENABLED:
            return DecisionResult(
                action="LIVE_READY" if live_mode else "PAPER_TRADE",
                live_ready=live_mode,
                data_quality_score=features.data_quality_score,
                reasons=["decision engine disabled"],
                feature_snapshot=features.compact(),
            )

        if not signal.is_actionable:
            return DecisionResult(
                action="SKIP",
                live_ready=False,
                data_quality_score=features.data_quality_score,
                reasons=["core EV/liquidity filters failed"],
                feature_snapshot=features.compact(),
            )

        reasons = ["core EV/liquidity filters passed"]
        warnings: list[str] = []
        risk_multiplier = 1.0

        if features.missing_sources:
            warnings.append("missing " + ",".join(sorted(set(features.missing_sources))))
        if features.stale_sources:
            warnings.append("stale " + ",".join(sorted(set(features.stale_sources))))

        if features.data_quality_score < DECISION_MIN_LIVE_FEATURE_QUALITY:
            risk_multiplier *= DECISION_LOW_DATA_RISK_MULTIPLIER
            warnings.append(
                f"feature quality {features.data_quality_score:.2f} below live threshold "
                f"{DECISION_MIN_LIVE_FEATURE_QUALITY:.2f}"
            )

        concentration = features.top_holder_concentration
        if concentration is not None and concentration >= DECISION_MAX_HOLDER_CONCENTRATION:
            return DecisionResult(
                action="WATCH",
                live_ready=False,
                risk_multiplier=0.0,
                data_quality_score=features.data_quality_score,
                reasons=[
                    f"top-holder concentration {concentration:.2f} exceeds "
                    f"{DECISION_MAX_HOLDER_CONCENTRATION:.2f}"
                ],
                warnings=warnings,
                feature_snapshot=features.compact(),
            )
        if concentration is not None and concentration >= DECISION_MAX_HOLDER_CONCENTRATION * 0.75:
            risk_multiplier *= 0.75
            warnings.append(f"elevated holder concentration {concentration:.2f}")

        volatility = features.realized_volatility_24h
        if volatility is not None and volatility >= DECISION_MAX_VOLATILITY_24H:
            risk_multiplier *= 0.70
            warnings.append(f"high 24h volatility {volatility:.2f}")

        momentum = features.price_momentum_24h
        if momentum is not None:
            if signal.direction == "YES" and momentum <= -0.25:
                return self._watch("strong adverse YES momentum", features, warnings)
            if signal.direction == "NO" and momentum >= 0.25:
                return self._watch("strong adverse NO momentum", features, warnings)
            if signal.direction == "YES" and momentum <= -0.10:
                risk_multiplier *= 0.80
                warnings.append(f"adverse YES momentum {momentum:.2f}")
            if signal.direction == "NO" and momentum >= 0.10:
                risk_multiplier *= 0.80
                warnings.append(f"adverse NO momentum {momentum:.2f}")

        imbalance = features.trade_imbalance_1h
        if imbalance is not None:
            if signal.direction == "YES" and imbalance <= -DECISION_CONTRA_FLOW_THRESHOLD:
                risk_multiplier *= 0.75
                warnings.append(f"contra flow against YES {imbalance:.2f}")
            if signal.direction == "NO" and imbalance >= DECISION_CONTRA_FLOW_THRESHOLD:
                risk_multiplier *= 0.75
                warnings.append(f"contra flow against NO {imbalance:.2f}")

        risk_multiplier = max(0.0, min(1.0, risk_multiplier))
        if risk_multiplier <= 0:
            return DecisionResult(
                action="SKIP",
                live_ready=False,
                risk_multiplier=0.0,
                data_quality_score=features.data_quality_score,
                reasons=["risk multiplier reached zero"],
                warnings=warnings,
                feature_snapshot=features.compact(),
            )

        live_ready = features.data_quality_score >= DECISION_MIN_LIVE_FEATURE_QUALITY and not features.stale_sources
        action = "LIVE_READY" if live_mode and live_ready else "PAPER_TRADE"
        if live_mode and not live_ready:
            reasons.append("not live-ready; paper/demo only until feature quality improves")
        return DecisionResult(
            action=action,
            live_ready=live_ready,
            risk_multiplier=round(risk_multiplier, 4),
            data_quality_score=features.data_quality_score,
            reasons=reasons,
            warnings=warnings,
            feature_snapshot=features.compact(),
        )

    def _watch(
        self,
        reason: str,
        features: MarketFeatures,
        warnings: list[str],
    ) -> DecisionResult:
        return DecisionResult(
            action="WATCH",
            live_ready=False,
            risk_multiplier=0.0,
            data_quality_score=features.data_quality_score,
            reasons=[reason],
            warnings=warnings,
            feature_snapshot=features.compact(),
        )
