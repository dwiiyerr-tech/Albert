"""
Probability Calibration — learns how to correct the agent's raw probability
estimates using Brier score decomposition and isotonic regression.

A well-calibrated agent that says "70% YES" should win ~70% of the time.
This module measures and corrects over/under-confidence systematically,
city-by-city and globally.

The calibrated probability replaces the raw simulation consensus before it
reaches the EV calculator, making every subsequent decision sharper.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

from learning.memory import ExperienceMemory, PredictionRecord

logger = logging.getLogger(__name__)

# Minimum resolved predictions before calibration kicks in
MIN_SAMPLES_GLOBAL = 20
MIN_SAMPLES_CITY = 10

# Use wider bins (0.10) when data is sparse, narrow (0.05) once we have ≥100 global samples.
# Wider bins fill faster and reduce noise from sparse data.
_WIDE_BIN_EDGES = [i / 10 for i in range(11)]   # 0.0, 0.1, ..., 1.0  (10 bins)
_NARROW_BIN_EDGES = [i / 20 for i in range(21)] # 0.0, 0.05, ..., 1.0 (20 bins)
NARROW_BIN_THRESHOLD = 100  # switch to narrow bins after this many global samples

BIN_EDGES = _NARROW_BIN_EDGES  # kept for external reference; actual edges chosen dynamically


def _choose_edges(n_global: int) -> list[float]:
    return _NARROW_BIN_EDGES if n_global >= NARROW_BIN_THRESHOLD else _WIDE_BIN_EDGES


def _bin_index(p: float, edges: list[float]) -> int:
    n_bins = len(edges) - 1
    step = edges[1] - edges[0]
    return min(int(p / step), n_bins - 1)


class CalibrationBin:
    """Tracks predicted vs actual in a probability band."""
    def __init__(self, lo: float, hi: float) -> None:
        self.lo = lo
        self.hi = hi
        self.predicted_sum = 0.0
        self.actual_sum = 0.0
        self.count = 0

    def add(self, predicted: float, actual: float) -> None:
        self.predicted_sum += predicted
        self.actual_sum += actual
        self.count += 1

    @property
    def mean_predicted(self) -> Optional[float]:
        return self.predicted_sum / self.count if self.count else None

    @property
    def mean_actual(self) -> Optional[float]:
        return self.actual_sum / self.count if self.count else None

    @property
    def calibration_error(self) -> Optional[float]:
        """Signed error: positive = overconfident, negative = underconfident."""
        if self.mean_predicted is None or self.mean_actual is None:
            return None
        return self.mean_predicted - self.mean_actual


class ProbabilityCalibrator:
    """
    Fits a per-bin calibration correction from historical predictions and
    applies it at inference time. Falls back to global calibration when
    city-specific data is sparse.

    Also computes:
    - Mean Brier Score (overall prediction quality)
    - Resolution (how spread out are predictions?)
    - Reliability (how close are predictions to actual frequencies?)
    - Expected Calibration Error (ECE)
    """

    def __init__(self, memory: ExperienceMemory) -> None:
        self._memory = memory
        self._global_bins: list[CalibrationBin] = []
        self._city_bins: dict[str, list[CalibrationBin]] = {}
        self._active_edges: list[float] = _WIDE_BIN_EDGES
        self._fitted = False

    def _make_bins(self, edges: list[float]) -> list[CalibrationBin]:
        return [CalibrationBin(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    # ─── Fitting ──────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """
        Refit calibration bins from all resolved predictions in memory.
        Uses wider bins when data is sparse (< 100 global samples) to ensure
        bins are populated enough to give meaningful corrections.
        """
        resolved = self._memory.resolved_records(last_n=500)
        valid = [r for r in resolved if r.outcome_yes is not None]
        if not valid:
            logger.debug("Calibrator: no resolved predictions yet, skipping fit")
            return

        # Choose bin width based on data volume
        edges = _choose_edges(len(valid))
        self._active_edges = edges

        self._global_bins = self._make_bins(edges)
        self._city_bins = {}

        for rec in valid:
            p = rec.consensus_probability
            actual = 1.0 if rec.outcome_yes else 0.0
            idx = _bin_index(p, edges)

            self._global_bins[idx].add(p, actual)

            city = rec.city
            if city not in self._city_bins:
                self._city_bins[city] = self._make_bins(edges)
            self._city_bins[city][idx].add(p, actual)

        self._fitted = True
        bin_width = edges[1] - edges[0]
        logger.info(
            "Calibrator fitted: %d predictions, %d cities, bin_width=%.2f",
            len(valid), len(self._city_bins), bin_width,
        )

    # ─── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, raw_p: float, city: Optional[str] = None) -> float:
        """
        Return a calibrated probability given a raw simulation output.
        Uses city-specific bins when data is sufficient, falls back to global.
        """
        if not self._fitted:
            self.fit()

        if not self._global_bins:
            return raw_p  # no data at all yet

        edges = self._active_edges
        idx = _bin_index(raw_p, edges)

        # Try city-specific first
        if city and city in self._city_bins:
            city_bin = self._city_bins[city][idx]
            city_global_count = sum(b.count for b in self._city_bins[city])
            if city_global_count >= MIN_SAMPLES_CITY and city_bin.count >= 2:
                correction = city_bin.calibration_error or 0.0
                calibrated = max(0.02, min(0.98, raw_p - correction))
                logger.debug(
                    "City calibration %s: raw=%.3f correction=%+.3f → %.3f",
                    city, raw_p, correction, calibrated,
                )
                return calibrated

        # Fall back to global
        global_count = sum(b.count for b in self._global_bins)
        if global_count >= MIN_SAMPLES_GLOBAL:
            global_bin = self._global_bins[idx]
            if global_bin.count >= 2:
                correction = global_bin.calibration_error or 0.0
                calibrated = max(0.02, min(0.98, raw_p - correction))
                logger.debug(
                    "Global calibration: raw=%.3f correction=%+.3f → %.3f",
                    raw_p, correction, calibrated,
                )
                return calibrated

        # Not enough data yet — return raw (log only once per session)
        logger.debug("Calibration: insufficient data (n=%d), returning raw p=%.3f",
                     global_count, raw_p)
        return raw_p

    # ─── Diagnostics ──────────────────────────────────────────────────────────

    def brier_score(self, city: Optional[str] = None) -> Optional[float]:
        """Mean Brier score — lower is better (0 = perfect, 1 = worst)."""
        records = self._memory.resolved_records(city=city, last_n=200)
        scores = [r.brier_score for r in records if r.brier_score is not None]
        return sum(scores) / len(scores) if scores else None

    def expected_calibration_error(self, city: Optional[str] = None) -> Optional[float]:
        """ECE — weighted average calibration error across bins."""
        if not self._fitted:
            self.fit()
        bins = self._city_bins.get(city, self._global_bins) if city else self._global_bins
        total_count = sum(b.count for b in bins)
        if total_count == 0:
            return None
        ece = sum(
            (b.count / total_count) * abs(b.calibration_error or 0.0)
            for b in bins
        )
        return ece

    def calibration_report(self) -> str:
        """Human-readable calibration report for logging / display."""
        if not self._fitted:
            self.fit()

        lines = ["=== Calibration Report ==="]
        global_brier = self.brier_score()
        global_ece = self.expected_calibration_error()
        lines.append(f"Global Brier Score: {global_brier:.4f}" if global_brier else "Global Brier Score: N/A")
        lines.append(f"Global ECE:         {global_ece:.4f}" if global_ece else "Global ECE: N/A")
        lines.append("")
        lines.append(f"{'Bin':>8}  {'Count':>5}  {'MeanPred':>9}  {'MeanActual':>10}  {'Error':>8}")
        lines.append("─" * 55)
        for b in self._global_bins:
            if b.count > 0:
                mp = b.mean_predicted or 0
                ma = b.mean_actual or 0
                err = b.calibration_error or 0
                lines.append(
                    f"  {b.lo:.2f}–{b.hi:.2f}  {b.count:>5}  {mp:>9.3f}  {ma:>10.3f}  {err:>+8.3f}"
                )

        lines.append("")
        lines.append("City breakdown:")
        for city, bins in sorted(self._city_bins.items()):
            n = sum(b.count for b in bins)
            if n < 5:
                continue
            brier = self.brier_score(city)
            ece = self.expected_calibration_error(city)
            lines.append(
                f"  {city:<18} n={n:>3}  "
                f"Brier={brier:.4f if brier else '  N/A'}  "
                f"ECE={ece:.4f if ece else '  N/A'}"
            )
        return "\n".join(lines)

    def city_overconfidence_flags(self) -> list[dict]:
        """
        Return cities where the agent is systematically over/under-confident.
        Used by the reflection engine to trigger persona evolution.
        """
        if not self._fitted:
            self.fit()
        flags = []
        for city, bins in self._city_bins.items():
            n = sum(b.count for b in bins)
            if n < MIN_SAMPLES_CITY:
                continue
            weighted_err = sum(
                (b.count / n) * (b.calibration_error or 0.0)
                for b in bins
            )
            if abs(weighted_err) > 0.08:
                flags.append({
                    "city": city,
                    "mean_calibration_error": round(weighted_err, 4),
                    "direction": "overconfident" if weighted_err > 0 else "underconfident",
                    "n": n,
                })
        return sorted(flags, key=lambda x: -abs(x["mean_calibration_error"]))
