"""
Weather Data Layer — inspired by WeatherBot (alteregoeth-ai/weatherbot)
Fetches forecasts from Open-Meteo (ECMWF + GFS/HRRR models) and
live METAR observations from Aviation Weather.
"""
from __future__ import annotations

import datetime
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import (
    AVIATION_WEATHER_BASE,
    OPEN_METEO_BASE,
    VISUAL_CROSSING_API_KEY,
    HIGH_SPREAD_THRESHOLD_F,
)

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _retry_get(url: str, params: dict, retries: int = 3,
               backoff: tuple = (2, 4, 8), timeout: int = 10) -> Optional[dict]:
    """HTTP GET with exponential backoff retry."""
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        wait = backoff[min(attempt - 1, len(backoff) - 1)] if backoff else 0
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt < attempts:
                logger.warning("HTTP attempt %d/%d failed for %s: %s — retrying in %ds",
                               attempt, attempts, url, exc, wait)
                if wait > 0:
                    time.sleep(wait)
            else:
                logger.warning("HTTP request failed after %d attempts: %s | %s", attempts, url, exc)
    return None


# Keep old name as alias for backward compat
def _safe_get(url: str, params: dict, timeout: int = 10) -> Optional[dict]:
    return _retry_get(url, params, retries=3, timeout=timeout)


@dataclass
class TemperatureReading:
    source: str
    city: str
    date: datetime.date
    temp_c: float
    temp_f: float = field(init=False)
    confidence: float = 1.0  # 0–1 model confidence weight

    def __post_init__(self) -> None:
        self.temp_f = self.temp_c * 9 / 5 + 32


@dataclass
class CityForecast:
    city: str
    lat: float
    lon: float
    ecmwf: Optional[TemperatureReading] = None
    gfs: Optional[TemperatureReading] = None
    metar: Optional[TemperatureReading] = None

    @property
    def consensus_temp_f(self) -> Optional[float]:
        """Weighted average across available models."""
        readings = [r for r in [self.ecmwf, self.gfs, self.metar] if r is not None]
        if not readings:
            return None
        total_weight = sum(r.confidence for r in readings)
        if total_weight <= 0:
            return sum(r.temp_f for r in readings) / len(readings)
        return sum(r.temp_f * r.confidence for r in readings) / total_weight

    @property
    def model_spread_f(self) -> Optional[float]:
        """Spread between highest and lowest model estimate."""
        readings = [r for r in [self.ecmwf, self.gfs, self.metar] if r is not None]
        if len(readings) < 2:
            return None
        temps = [r.temp_f for r in readings]
        return max(temps) - min(temps)

    def as_dict(self) -> dict:
        return {
            "city": self.city,
            "consensus_temp_f": self.consensus_temp_f,
            "model_spread_f": self.model_spread_f,
            "ecmwf_f": self.ecmwf.temp_f if self.ecmwf else None,
            "gfs_f": self.gfs.temp_f if self.gfs else None,
            "metar_f": self.metar.temp_f if self.metar else None,
        }


def get_ecmwf_forecast(lat: float, lon: float, city: str, target_date: datetime.date) -> Optional[TemperatureReading]:
    """Fetch ECMWF IFS forecast via Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "celsius",
        "forecast_days": 7,
        "models": "ecmwf_ifs025",
        "timezone": "auto",
    }
    data = _safe_get(f"{OPEN_METEO_BASE}/forecast", params)
    if not data:
        return None
    try:
        dates = [datetime.date.fromisoformat(d) for d in data["daily"]["time"]]
        temps = data["daily"]["temperature_2m_max"]
        idx = dates.index(target_date)
        raw_c = temps[idx]
        # ECMWF tends to run ~0.3°C warm in summer; apply small bias correction
        corrected_c = raw_c - 0.3
        return TemperatureReading(source="ECMWF", city=city, date=target_date,
                                  temp_c=corrected_c, confidence=0.9)
    except (KeyError, ValueError, IndexError):
        return None


def get_gfs_forecast(lat: float, lon: float, city: str, target_date: datetime.date) -> Optional[TemperatureReading]:
    """Fetch GFS/HRRR forecast via Open-Meteo (best_match model)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "celsius",
        "forecast_days": 7,
        "models": "gfs_seamless",
        "timezone": "auto",
    }
    data = _safe_get(f"{OPEN_METEO_BASE}/forecast", params)
    if not data:
        return None
    try:
        dates = [datetime.date.fromisoformat(d) for d in data["daily"]["time"]]
        temps = data["daily"]["temperature_2m_max"]
        idx = dates.index(target_date)
        return TemperatureReading(source="GFS", city=city, date=target_date,
                                  temp_c=temps[idx], confidence=0.85)
    except (KeyError, ValueError, IndexError):
        return None


def get_metar_observation(icao: str, city: str) -> Optional[TemperatureReading]:
    """Fetch latest METAR temperature observation from Aviation Weather."""
    params = {"ids": icao, "format": "json"}
    data = _safe_get(f"{AVIATION_WEATHER_BASE}/metar", params)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    try:
        obs = data[0]
        # Try multiple field names used by different Aviation Weather API versions
        raw = None
        for field_name in ("temp", "tmpf", "tempC", "tempF", "temperature"):
            raw = obs.get(field_name)
            if raw is not None:
                break
        if raw is None:
            logger.warning("METAR %s: no temperature field found in %s", icao, list(obs.keys()))
            return None
        raw_val = float(raw)
        # If field name contains 'f' (tmpf, tempF) the value is Fahrenheit — always convert
        if "f" in (field_name or "").lower():
            temp_c = (raw_val - 32) * 5 / 9
        else:
            temp_c = raw_val
        today = datetime.date.today()
        return TemperatureReading(source="METAR", city=city, date=today,
                                  temp_c=temp_c, confidence=0.95)
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning("METAR parse failed for %s: %s", icao, exc)
        return None


def get_historical_temp_openmeteo(lat: float, lon: float, city: str,
                                   date: datetime.date) -> Optional[TemperatureReading]:
    """
    Fetch historical temperature from Open-Meteo archive API.
    Free, no API key required. Used as primary fallback for prediction resolution.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
        "daily": "temperature_2m_max",
        "temperature_unit": "celsius",
        "timezone": "auto",
    }
    data = _safe_get(OPEN_METEO_ARCHIVE, params)
    if not data:
        return None
    try:
        temp_c = data["daily"]["temperature_2m_max"][0]
        if temp_c is None:
            return None
        return TemperatureReading(source="OpenMeteoArchive", city=city, date=date,
                                  temp_c=temp_c, confidence=0.98)
    except (KeyError, IndexError, TypeError):
        return None


def get_historical_temp(city: str, date: datetime.date,
                         lat: Optional[float] = None,
                         lon: Optional[float] = None) -> Optional[TemperatureReading]:
    """
    Fetch historical actuals.
    Tries Visual Crossing first (if API key set), then falls back to
    Open-Meteo archive (free, no key needed) using lat/lon.
    """
    # Primary: Visual Crossing
    if VISUAL_CROSSING_API_KEY:
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services"
            f"/timeline/{city}/{date.isoformat()}"
        )
        params = {
            "unitGroup": "metric",
            "key": VISUAL_CROSSING_API_KEY,
            "include": "days",
            "elements": "tempmax",
        }
        data = _safe_get(url, params)
        if data:
            try:
                temp_c = data["days"][0]["tempmax"]
                return TemperatureReading(source="VisualCrossing", city=city, date=date,
                                          temp_c=temp_c, confidence=1.0)
            except (KeyError, IndexError):
                pass

    # Fallback: Open-Meteo archive (free, no key)
    if lat is not None and lon is not None:
        return get_historical_temp_openmeteo(lat, lon, city, date)

    logger.warning("Cannot resolve historical temp for %s %s: no API key and no lat/lon", city, date)
    return None


def fetch_city_forecast(city_cfg: dict, target_date: datetime.date) -> CityForecast:
    """
    Aggregate all model forecasts for a city on a given date.
    Sets confidence to 'low' automatically when model spread exceeds HIGH_SPREAD_THRESHOLD_F.
    """
    name = city_cfg["name"]
    lat, lon = city_cfg["lat"], city_cfg["lon"]
    metar_id = city_cfg.get("metar")

    forecast = CityForecast(city=name, lat=lat, lon=lon)
    forecast.ecmwf = get_ecmwf_forecast(lat, lon, name, target_date)
    forecast.gfs = get_gfs_forecast(lat, lon, name, target_date)
    if metar_id and target_date == datetime.date.today():
        forecast.metar = get_metar_observation(metar_id, name)

    spread = forecast.model_spread_f or 0.0
    if spread > HIGH_SPREAD_THRESHOLD_F:
        logger.warning(
            "High model disagreement for %s %s: spread=%.1f°F > threshold=%.1f°F — "
            "simulation will be flagged low-confidence",
            name, target_date, spread, HIGH_SPREAD_THRESHOLD_F,
        )

    logger.info(
        "Forecast %s %s → consensus=%.1f°F spread=%.1f°F%s",
        name, target_date,
        forecast.consensus_temp_f or 0,
        spread,
        " [HIGH DISAGREEMENT]" if spread > HIGH_SPREAD_THRESHOLD_F else "",
    )
    return forecast


def parse_temp_range(question: str) -> Optional[tuple[float, float]]:
    """Extract (low, high) °F bounds from a Polymarket market question string."""
    pattern = r"([-\d]+)\s*(?:-|–|to|and)\s*([-\d]+)\s*(?:°?\s*[Ff]|degrees?)?"
    match = re.search(pattern, question)
    if match:
        return float(match.group(1)), float(match.group(2))
    # single-threshold markets: "above 95°F", "over 80 degrees", "at least 70"
    above = re.search(
        r"(?:above|over|at\s+least|greater\s+than|exceed(?:s|ing)?)\s+([-\d]+)\s*(?:°?\s*[Ff]|degrees?)?",
        question,
        re.IGNORECASE,
    )
    if above:
        v = float(above.group(1))
        return v, float("inf")
    # "80°F or higher"
    higher = re.search(r"([-\d]+)\s*(?:°?\s*[Ff]|degrees?)?\s+or\s+(?:higher|above|more)", question, re.IGNORECASE)
    if higher:
        v = float(higher.group(1))
        return v, float("inf")
    below = re.search(
        r"(?:below|under|less\s+than|at\s+most|no\s+more\s+than)\s+([-\d]+)\s*(?:°?\s*[Ff]|degrees?)?",
        question,
        re.IGNORECASE,
    )
    if below:
        v = float(below.group(1))
        return float("-inf"), v
    lower = re.search(r"([-\d]+)\s*(?:°?\s*[Ff]|degrees?)?\s+or\s+(?:lower|below|less)", question, re.IGNORECASE)
    if lower:
        v = float(lower.group(1))
        return float("-inf"), v
    return None
