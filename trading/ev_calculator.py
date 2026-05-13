"""
Expected Value & Kelly Criterion Calculator — from WeatherBot (alteregoeth-ai/weatherbot)

Extended to accept simulation-derived probabilities instead of raw model output,
giving the multi-agent consensus a direct path to trade sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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
from simulation.agents import SimulationResult


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
    """

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
        """Return fractional Kelly stake (capped at KELLY_FRACTION)."""
        if price <= 0 or price >= 1:
            return 0.0
        b = (1.0 / price) - 1.0
        kelly = (probability * b - (1.0 - probability)) / b
        kelly = max(0.0, kelly)
        return min(kelly * KELLY_FRACTION, KELLY_FRACTION)

    def evaluate(
        self,
        sim: SimulationResult,
        market_price: float,
        market_id: str,
        hours_to_resolution: float,
        volume: float,
        spread: float,
    ) -> Optional[TradeSignal]:
        """
        Generate a TradeSignal from a SimulationResult and live market data.
        Returns None if the market doesn't meet basic filters.
        """
        # Basic market quality filters (from WeatherBot entry conditions)
        if spread > MAX_SPREAD:
            return None
        if volume < MIN_VOLUME:
            return None
        if not (MIN_HOURS_TO_RESOLUTION <= hours_to_resolution <= MAX_HOURS_TO_RESOLUTION):
            return None

        p_yes = sim.consensus_probability

        # Try both YES and NO directions
        ev_yes = self.compute_ev(p_yes, market_price)
        ev_no = self.compute_ev(1 - p_yes, 1 - market_price)

        if ev_yes >= ev_no and ev_yes >= 0:
            direction = "YES"
            ev = ev_yes
            probability = p_yes
            price = market_price
        elif ev_no > ev_yes and ev_no >= 0:
            direction = "NO"
            ev = ev_no
            probability = 1 - p_yes
            price = 1 - market_price
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
        )
