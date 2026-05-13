"""
Weather Knowledge Graph — inspired by MiroFish's GraphRAG-based seed extraction.

Builds a lightweight in-memory knowledge graph connecting cities, historical
temperature patterns, and forecast model performance records.
This graph informs the multi-agent debate with structured contextual memory.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import SIM_STATE_FILE

logger = logging.getLogger(__name__)


@dataclass
class ModelAccuracyRecord:
    """Tracks how accurate each model has been for a city."""
    city: str
    model: str
    predictions: list[float] = field(default_factory=list)  # predicted °F
    actuals: list[float] = field(default_factory=list)       # actual °F

    @property
    def mean_absolute_error(self) -> Optional[float]:
        if not self.predictions or len(self.predictions) != len(self.actuals):
            return None
        errors = [abs(p - a) for p, a in zip(self.predictions, self.actuals)]
        return sum(errors) / len(errors)

    @property
    def sample_size(self) -> int:
        return len(self.predictions)

    def record(self, predicted: float, actual: float) -> None:
        self.predictions.append(predicted)
        self.actuals.append(actual)


@dataclass
class CityNode:
    """A city entity in the knowledge graph with historical context."""
    name: str
    lat: float
    lon: float
    climate_zone: str = "unknown"
    # daily historical max temps for the last 30 days (°F)
    recent_temps_f: list[float] = field(default_factory=list)
    model_accuracy: dict[str, ModelAccuracyRecord] = field(default_factory=dict)

    @property
    def climatological_mean_f(self) -> Optional[float]:
        if not self.recent_temps_f:
            return None
        return sum(self.recent_temps_f) / len(self.recent_temps_f)

    @property
    def climatological_std_f(self) -> Optional[float]:
        if len(self.recent_temps_f) < 2:
            return None
        mean = self.climatological_mean_f
        variance = sum((t - mean) ** 2 for t in self.recent_temps_f) / len(self.recent_temps_f)
        return variance ** 0.5

    def best_model(self) -> Optional[str]:
        """Return the model with lowest MAE for this city."""
        scored = {
            m: r.mean_absolute_error
            for m, r in self.model_accuracy.items()
            if r.mean_absolute_error is not None and r.sample_size >= 5
        }
        if not scored:
            return None
        return min(scored, key=lambda m: scored[m])

    def update_model_accuracy(self, model: str, predicted: float, actual: float) -> None:
        if model not in self.model_accuracy:
            self.model_accuracy[model] = ModelAccuracyRecord(self.name, model)
        self.model_accuracy[model].record(predicted, actual)


class WeatherKnowledgeGraph:
    """
    In-memory knowledge graph that persists to disk as JSON.
    Tracks city climatology, inter-city correlations, and model accuracy.
    """

    def __init__(self, state_file: str = SIM_STATE_FILE) -> None:
        self._state_file = state_file
        self.cities: dict[str, CityNode] = {}
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                raw = json.load(f)
            for name, data in raw.get("cities", {}).items():
                node = CityNode(
                    name=data["name"],
                    lat=data["lat"],
                    lon=data["lon"],
                    climate_zone=data.get("climate_zone", "unknown"),
                    recent_temps_f=data.get("recent_temps_f", []),
                )
                for model, acc in data.get("model_accuracy", {}).items():
                    node.model_accuracy[model] = ModelAccuracyRecord(
                        city=name,
                        model=model,
                        predictions=acc.get("predictions", []),
                        actuals=acc.get("actuals", []),
                    )
                self.cities[name] = node
            logger.info("Knowledge graph loaded: %d cities", len(self.cities))
        except Exception as exc:
            logger.warning("Could not load knowledge graph: %s", exc)

    def save(self) -> None:
        payload: dict = {"cities": {}}
        for name, node in self.cities.items():
            payload["cities"][name] = {
                "name": node.name,
                "lat": node.lat,
                "lon": node.lon,
                "climate_zone": node.climate_zone,
                "recent_temps_f": node.recent_temps_f[-30:],  # keep last 30
                "model_accuracy": {
                    m: {"predictions": r.predictions[-60:], "actuals": r.actuals[-60:]}
                    for m, r in node.model_accuracy.items()
                },
            }
        with open(self._state_file, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug("Knowledge graph saved")

    # ─── Graph operations ─────────────────────────────────────────────────────

    def upsert_city(self, name: str, lat: float, lon: float) -> CityNode:
        if name not in self.cities:
            self.cities[name] = CityNode(name=name, lat=lat, lon=lon)
        return self.cities[name]

    def record_temperature(self, city: str, temp_f: float) -> None:
        if city in self.cities:
            self.cities[city].recent_temps_f.append(temp_f)

    def record_model_outcome(self, city: str, model: str,
                              predicted_f: float, actual_f: float) -> None:
        if city in self.cities:
            self.cities[city].update_model_accuracy(model, predicted_f, actual_f)

    def context_for_city(self, city: str) -> dict:
        """Return structured context for injection into agent prompts."""
        node = self.cities.get(city)
        if not node:
            return {}
        best = node.best_model()
        return {
            "climate_zone": node.climate_zone,
            "30d_mean_f": node.climatological_mean_f,
            "30d_std_f": node.climatological_std_f,
            "best_model": best,
            "best_model_mae": (
                node.model_accuracy[best].mean_absolute_error if best else None
            ),
        }

    def find_correlated_cities(self, city: str, top_n: int = 3) -> list[str]:
        """Return cities with similar recent temperature patterns (by lat proximity)."""
        node = self.cities.get(city)
        if not node:
            return []
        others = [
            (abs(n.lat - node.lat), n.name)
            for n_name, n in self.cities.items()
            if n_name != city
        ]
        others.sort()
        return [name for _, name in others[:top_n]]
