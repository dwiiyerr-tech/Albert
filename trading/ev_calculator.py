"""
Expected Value & Kelly Criterion Calculator — from WeatherBot (alteregoeth-ai/weatherbot)

Extended to accept simulation-derived probabilities instead of raw model output,
giving the multi-agent consensus a direct path to trade sizing.

Learning integration: the calibrator corrects raw simulation probabilities using
historical Brier score data, and market_learner adjusts for discovered city/timing
market inefficiencies before EV is computed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from config import (
    KELLY_FRACTION,
    MAX_TRADE_SIZE_USD,
    MIN_EV,
    MAX_SPREAD,
    MIN_VOLUME,
    MIN_HOURS_TO_RESOLUTION,
    MAX_HOURS_TO_RESOLUTION,
    CONSENSUS_THRESHOLD,
)

if TYPE_CHECKING:
    from learning.calibration import ProbabilityCalibrator
    from learning.market_learner import MarketPatternLearner
    from simulation.agents import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    city: str
    target_date: str
    bucket_low: float
    bucket_high: float
    direction: str          # "YES" or "NO"
    probability: float      # P(outcome=YES) from simulation
    market_price: float     # current market price (0–1)
    ev: float               # expected value per dollar risked
    kelly_fraction: float   # optimal fraction of bankroll
    recommended_usd: float  # dollar amount to wager
    confidence_level: str   # low / medium / high from simulation
    hours_to_resolution: float
    volume: float
    market_id: str = ""
    no_token_id: str = ""   # CLOB NO token ID (needed for live NO-direction orders)

    @property
    def is_actionable(self) -> bool:
        return (
            self.ev >= MIN_EV
            and self.confidence_level != "low"
            and self.hours_to_resolution >= MIN_HOURS_TO_RESOLUTION
            and self.hours_to_resolution <= MAX_HOURS_TO_RESOLUTION
            and self.volume >= MIN_VOLUME
        )


class EVCalculator:
    """
    Converts simulation consensus probabilities into tradeable signals
    using Expected Value and Kelly Criterion position sizing.

    EV formula (from WeatherBot):
        EV = P × (1/price − 1) − (1 − P)

    Kelly fraction:
        f = (P × b − (1−P)) / b,  b = (1/price − 1)

    Learning integration: pass calibrator and market_learner to automatically
    correct raw probabilities before EV computation.
    """

    def __init__(
        self,
        calibrator: Optional["ProbabilityCalibrator"] = None,
        market_learner: Optional["MarketPatternLearner"] = None,
    ) -> None:
        self._calibrator = calibrator
        self._market_learner = market_learner

    def _adjust_probability(
        self,
        raw_p: float,
        city: str,
        hours_to_resolution: float,
    ) -> tuple[float, list[str]]:
        """
        Apply learned corrections to a raw simulation probability.
        Fix: corrections are summed first, then applied once with a ±0.20 cap
        to prevent stacking from collapsing probabilities to extremes.
        Returns (adjusted_p, list_of_adjustments_applied).
        """
        total_delta = 0.0
        adjustments: list[str] = []

        # 1. Calibration correction (Brier-score-based)
        if self._calibrator:
            calibrated = self._calibrator.calibrate(raw_p, city)
            delta = calibrated - raw_p
            if abs(delta) > 0.005:
                total_delta += delta
                adjustments.append(f"calibration {delta:+.3f}")

        # 2. City market bias correction
        if self._market_learner:
            city_adj = self._market_learner.ev_adjustment_for_city(city)
            if abs(city_adj) > 0.005:
                total_delta -= city_adj
                adjustments.append(f"city_bias {-city_adj:+.3f}")

            timing_adj = self._market_learner.timing_adjustment(hours_to_resolution)
            if abs(timing_adj) > 0.005:
                total_delta -= timing_adj
                adjustments.append(f"timing {-timing_adj:+.3f}")

        # Apply combined correction with a ±0.20 cap to prevent extremes
        total_delta = max(-0.20, min(0.20, total_delta))
        p = max(0.02, min(0.98, raw_p + total_delta))
        return p, adjustments

    def compute_ev(self, probability: float, price: float) -> float:
        """
        Compute EV for buying a YES contract at `price`.
        `price` is the contract price in [0, 1] (e.g. 0.40 = 40¢).
        """
        if price <= 0 or price >= 1:
            return -1.0
        b = (1.0 / price) - 1.0  # net odds
        ev = probability * b - (1.0 - probability)
        return ev

    def compute_kelly(self, probability: float, price: float) -> float:
        """Return fractional Kelly stake (capped at KELLY_FRACTION).
        Fix: was incorrectly applying `kelly * KELLY_FRACTION` then capping at
        KELLY_FRACTION, which double-applied the fraction. Correct form: cap the
        raw kelly at KELLY_FRACTION."""
        if price <= 0 or price >= 1:
            return 0.0
        b = (1.0 / price) - 1.0
        kelly = (probability * b - (1.0 - probability)) / b
        kelly = max(0.0, kelly)
        return min(kelly, KELLY_FRACTION)  # cap at fractional Kelly limit

    def evaluate(
        self,
        sim: SimulationResult,
        market_price: float,
        market_id: str,
        hours_to_resolution: float,
        volume: float,
        spread: float,
        no_token_id: str = "",
        market_price_no: float | None = None,
    ) -> Optional[TradeSignal]:
        """
        Generate a TradeSignal from a SimulationResult and live market data.
        Applies learned probability corrections before computing EV.
        Returns None if the market doesn't meet basic filters.
        """
        # Basic market quality filters (from WeatherBot entry conditions)
        if spread > MAX_SPREAD:
            return None
        if volume < MIN_VOLUME:
            return None
        if not (MIN_HOURS_TO_RESOLUTION <= hours_to_resolution <= MAX_HOURS_TO_RESOLUTION):
            return None

        # Apply learned probability corrections
        raw_p = sim.consensus_probability
        if math.isnan(raw_p) or raw_p <= 0 or raw_p >= 1:
            logger.warning("Invalid consensus_probability %.4f for %s — skipping", raw_p, sim.city)
            return None
        p_yes, adjustments = self._adjust_probability(raw_p, sim.city, hours_to_resolution)
        if adjustments:
            import logging
            logging.getLogger(__name__).info(
                "Probability adjusted %s: %.3f→%.3f (%s)",
                sim.city, raw_p, p_yes, ", ".join(adjustments),
            )

        # Try both YES and NO directions
        ev_yes = self.compute_ev(p_yes, market_price)
        no_price = market_price_no if market_price_no is not None else 1 - market_price
        ev_no = self.compute_ev(1 - p_yes, no_price)

        if ev_yes >= ev_no and ev_yes >= 0:
            direction = "YES"
            ev = ev_yes
            probability = p_yes
            price = market_price
        elif ev_no > ev_yes and ev_no >= 0:
            direction = "NO"
            ev = ev_no
            probability = 1 - p_yes
            price = no_price
        else:
            return None  # no positive EV on either side

        kelly = self.compute_kelly(probability, price)
        # Conservative position size: Kelly × $100 bankroll baseline, cap at MAX_TRADE_SIZE_USD
        recommended = min(kelly * 100.0, MAX_TRADE_SIZE_USD)

        return TradeSignal(
            city=sim.city,
            target_date=sim.target_date,
            bucket_low=sim.bucket_low,
            bucket_high=sim.bucket_high,
            direction=direction,
            probability=probability,
            market_price=price,
            ev=ev,
            kelly_fraction=kelly,
            recommended_usd=recommended,
            confidence_level=sim.confidence_level,
            hours_to_resolution=hours_to_resolution,
            volume=volume,
            market_id=market_id,
            no_token_id=no_token_id,
        )
