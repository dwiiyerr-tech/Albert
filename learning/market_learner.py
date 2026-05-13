"""
Market Pattern Learner — continuously mines Polymarket data for structural
patterns the agent can exploit: pricing inefficiencies, volume anomalies,
timing effects, city-specific market quirks, and liquidity traps.

All discovered patterns become Lessons in ExperienceMemory and are
automatically injected into the simulation debate and EV calculations.
"""
from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from learning.memory import ExperienceMemory, Lesson, MarketObservation
import uuid

logger = logging.getLogger(__name__)


@dataclass
class MarketEdge:
    """A quantified structural edge in the market."""
    name: str
    city: Optional[str]
    description: str
    mean_edge: float          # avg market mispricing direction (+/-) in probability units
    hit_rate: float           # fraction of time this pattern produces a correct signal
    sample_size: int
    actionable_rule: str

    @property
    def is_significant(self) -> bool:
        # Require at least 10 samples for basic significance
        return self.sample_size >= 10 and abs(self.mean_edge) > 0.05 and self.hit_rate > 0.55

    @property
    def statistical_confidence(self) -> float:
        """
        Approximate 95% confidence lower bound on hit_rate using Wilson score interval.
        More statistically grounded than ad-hoc formula.
        """
        if self.sample_size < 2:
            return 0.0
        n = self.sample_size
        p = self.hit_rate
        z = 1.96  # 95% CI
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, center - half_width)  # lower bound


class MarketPatternLearner:
    """
    Statistically analyses accumulated MarketObservations to find repeatable edges.
    Runs after each batch of market resolutions and persists findings as Lessons.
    """

    def __init__(self, memory: ExperienceMemory) -> None:
        self._memory = memory

    # ─── Pattern detection methods ────────────────────────────────────────────

    def _timing_effect(self, observations: list[MarketObservation]) -> list[MarketEdge]:
        """
        Does market accuracy vary with time-to-resolution?
        Markets far from resolution may be persistently mispriced.
        """
        resolved = [o for o in observations if o.resolved_yes is not None]
        if len(resolved) < 15:
            return []

        buckets: dict[str, list[float]] = {
            "2-12h": [], "12-24h": [], "24-48h": [], "48-72h": [],
        }
        for obs in resolved:
            h = obs.hours_to_resolution
            err = obs.price_yes - (1.0 if obs.resolved_yes else 0.0)
            if 2 <= h < 12:
                buckets["2-12h"].append(err)
            elif 12 <= h < 24:
                buckets["12-24h"].append(err)
            elif 24 <= h < 48:
                buckets["24-48h"].append(err)
            elif 48 <= h <= 72:
                buckets["48-72h"].append(err)

        edges = []
        for window, errors in buckets.items():
            if len(errors) < 5:
                continue
            mean_err = sum(errors) / len(errors)
            hit_rate = sum(1 for e in errors if (e > 0) == (mean_err > 0)) / len(errors)
            if abs(mean_err) > 0.05:
                direction = "overpriced" if mean_err > 0 else "underpriced"
                edges.append(MarketEdge(
                    name=f"timing_{window}",
                    city=None,
                    description=f"Markets with {window} to resolution are {direction} by {abs(mean_err):.1%}",
                    mean_edge=mean_err,
                    hit_rate=hit_rate,
                    sample_size=len(errors),
                    actionable_rule=(
                        f"When entering with {window} to resolution, "
                        f"{'buy NO' if mean_err > 0 else 'buy YES'} "
                        f"if price confirms this bias"
                    ),
                ))
        return edges

    def _volume_effect(self, observations: list[MarketObservation]) -> list[MarketEdge]:
        """
        Are thin-volume markets more mispriced?
        """
        resolved = [o for o in observations if o.resolved_yes is not None]
        if len(resolved) < 20:
            return []

        volumes = sorted(o.volume for o in resolved)
        median_vol = volumes[len(volumes) // 2]
        thin = [o for o in resolved if o.volume < median_vol]
        thick = [o for o in resolved if o.volume >= median_vol]

        edges = []
        for label, group in [("thin (<median volume)", thin), ("thick (≥median volume)", thick)]:
            if len(group) < 5:
                continue
            errors = [o.price_yes - (1.0 if o.resolved_yes else 0.0) for o in group]
            mean_err = sum(errors) / len(errors)
            abs_errors = [abs(e) for e in errors]
            mean_abs_err = sum(abs_errors) / len(abs_errors)
            if mean_abs_err > 0.08:
                edges.append(MarketEdge(
                    name=f"volume_{label.split()[0]}",
                    city=None,
                    description=(
                        f"{label.capitalize()} markets have higher average pricing error "
                        f"({mean_abs_err:.1%} MAE, mean bias {mean_err:+.1%})"
                    ),
                    mean_edge=mean_err,
                    hit_rate=0.6,
                    sample_size=len(group),
                    actionable_rule=(
                        f"In {label} markets, apply an extra 5% discount "
                        f"to market prices before comparison with simulation"
                    ),
                ))
        return edges

    def _city_market_bias(self, observations: list[MarketObservation]) -> list[MarketEdge]:
        """
        Are certain cities' markets persistently biased in one direction?
        """
        city_obs: dict[str, list[MarketObservation]] = defaultdict(list)
        for o in observations:
            if o.resolved_yes is not None:
                city_obs[o.city].append(o)

        edges = []
        for city, obs_list in city_obs.items():
            if len(obs_list) < 8:
                continue
            errors = [o.price_yes - (1.0 if o.resolved_yes else 0.0) for o in obs_list]
            mean_err = sum(errors) / len(errors)
            hit_rate = sum(1 for e in errors if (e > 0) == (mean_err > 0)) / len(errors)
            if abs(mean_err) > 0.07:
                direction = "systematically overpriced" if mean_err > 0 else "systematically underpriced"
                edges.append(MarketEdge(
                    name=f"city_bias_{city.lower().replace(' ', '_')}",
                    city=city,
                    description=(
                        f"{city} temperature markets are {direction} "
                        f"by {abs(mean_err):.1%} on average (n={len(obs_list)})"
                    ),
                    mean_edge=mean_err,
                    hit_rate=hit_rate,
                    sample_size=len(obs_list),
                    actionable_rule=(
                        f"For {city}: shift fair-value probability by "
                        f"{-mean_err:+.1%} to correct for market bias"
                    ),
                ))
        return edges

    def _spread_signal(self, observations: list[MarketObservation]) -> list[MarketEdge]:
        """
        Does bid-ask spread predict subsequent mispricing?
        Wide spreads may indicate uncertainty the market hasn't resolved.
        """
        resolved = [o for o in observations if o.resolved_yes is not None]
        if len(resolved) < 20:
            return []

        spreads = sorted(o.spread for o in resolved)
        p75 = spreads[int(len(spreads) * 0.75)]

        wide_spread = [o for o in resolved if o.spread >= p75]
        narrow_spread = [o for o in resolved if o.spread < p75]

        edges = []
        for label, group in [("wide-spread", wide_spread), ("narrow-spread", narrow_spread)]:
            if len(group) < 5:
                continue
            abs_errors = [abs(o.price_yes - (1.0 if o.resolved_yes else 0.0)) for o in group]
            mae = sum(abs_errors) / len(abs_errors)
            if label == "wide-spread" and mae > 0.12:
                edges.append(MarketEdge(
                    name="wide_spread_uncertainty",
                    city=None,
                    description=(
                        f"Wide-spread markets (≥{p75:.3f}) have {mae:.1%} MAE — "
                        f"high uncertainty; our simulation has more edge here"
                    ),
                    mean_edge=0.0,
                    hit_rate=0.65,
                    sample_size=len(group),
                    actionable_rule=(
                        "In wide-spread markets, increase recommended position size "
                        "by 20% when simulation confidence is 'high'"
                    ),
                ))
        return edges

    def _resolution_clustering(self, observations: list[MarketObservation]) -> list[MarketEdge]:
        """
        Are resolutions clustered in certain temperature bands?
        E.g. extreme heat buckets rarely resolve YES.
        """
        resolved = [o for o in observations if o.resolved_yes is not None]
        bucket_outcomes: dict[tuple, list[bool]] = defaultdict(list)
        for o in resolved:
            key = (round(o.bucket_low / 5) * 5, round(o.bucket_high / 5) * 5)
            bucket_outcomes[key].append(o.resolved_yes)

        edges = []
        for (lo, hi), outcomes in bucket_outcomes.items():
            if len(outcomes) < 8:
                continue
            yes_rate = sum(outcomes) / len(outcomes)
            # Extreme buckets
            if lo >= 100 or hi <= 30:
                if yes_rate < 0.15:
                    edges.append(MarketEdge(
                        name=f"extreme_bucket_{lo}_{hi}",
                        city=None,
                        description=(
                            f"Temperature bucket {lo}–{hi}°F resolves YES only {yes_rate:.0%} of the time. "
                            f"Market frequently misprices tail events."
                        ),
                        mean_edge=0.0,
                        hit_rate=1 - yes_rate,
                        sample_size=len(outcomes),
                        actionable_rule=(
                            f"For bucket {lo}–{hi}°F: default simulation p should be anchored "
                            f"toward {yes_rate:.0%} unless forecast strongly disagrees"
                        ),
                    ))
        return edges

    # ─── Main analysis ────────────────────────────────────────────────────────

    def analyze(self) -> list[Lesson]:
        """
        Run all pattern detectors and persist significant edges as Lessons.
        Returns list of newly discovered lessons.
        """
        observations = self._memory.observations
        if len(observations) < 15:
            logger.info("Not enough market observations for pattern analysis (%d)", len(observations))
            return []

        all_edges: list[MarketEdge] = []
        all_edges.extend(self._timing_effect(observations))
        all_edges.extend(self._volume_effect(observations))
        all_edges.extend(self._city_market_bias(observations))
        all_edges.extend(self._spread_signal(observations))
        all_edges.extend(self._resolution_clustering(observations))

        new_lessons: list[Lesson] = []
        for edge in all_edges:
            if not edge.is_significant:
                continue
            lesson = Lesson(
                id=str(uuid.uuid4()),
                category="market_pattern",
                city=edge.city,
                content=(
                    f"[edge_name={edge.name}] "
                    f"[mean_edge={edge.mean_edge:.4f}] "
                    f"[window={edge.name}] "
                    f"{edge.description} | "
                    f"Hit rate: {edge.hit_rate:.0%}, n={edge.sample_size}. "
                    f"Rule: {edge.actionable_rule}"
                ),
                # Use Wilson-score lower bound for confidence (statistically grounded)
                confidence=min(0.90, edge.statistical_confidence),
                supporting_records=[],
            )
            self._memory.add_lesson(lesson)
            new_lessons.append(lesson)

        logger.info(
            "Market pattern analysis: %d edges found, %d new lessons stored",
            len(all_edges), len(new_lessons),
        )
        return new_lessons

    @staticmethod
    def _extract_mean_edge(lesson_content: str) -> Optional[float]:
        """
        Extract the numeric mean_edge stored directly in the structured
        lesson content tag: [mean_edge=0.0712]. No fragile text parsing needed.
        """
        import re
        match = re.search(r"\[mean_edge=([+-]?\d+\.\d+)\]", lesson_content)
        if match:
            return float(match.group(1))
        return None

    def ev_adjustment_for_city(self, city: str) -> float:
        """
        Return a probability adjustment (delta) to apply for a given city.
        Reads mean_edge directly from the structured tag in lesson content.
        Positive = market overprices YES → we should discount YES probability.
        """
        city_lessons = [
            l for l in self._memory.lessons
            if l.city == city and l.category == "market_pattern"
            and "city_bias" in l.content
        ]
        if not city_lessons:
            return 0.0
        best = max(city_lessons, key=lambda l: l.reliability_score)
        return self._extract_mean_edge(best.content) or 0.0

    def timing_adjustment(self, hours_to_resolution: float) -> float:
        """
        Return a probability adjustment for a given time-to-resolution.
        Reads mean_edge directly from the structured tag in lesson content.
        """
        if hours_to_resolution < 2:
            window = "2-12h"
        elif hours_to_resolution < 12:
            window = "2-12h"
        elif hours_to_resolution < 24:
            window = "12-24h"
        elif hours_to_resolution < 48:
            window = "24-48h"
        else:
            window = "48-72h"

        timing_lessons = [
            l for l in self._memory.lessons
            if l.category == "market_pattern" and f"timing_{window}" in l.content
        ]
        if not timing_lessons:
            return 0.0
        best = max(timing_lessons, key=lambda l: l.reliability_score)
        return self._extract_mean_edge(best.content) or 0.0
