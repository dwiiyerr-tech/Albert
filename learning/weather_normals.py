from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _c_to_f(value: float) -> float:
    return value * 9 / 5 + 32


@dataclass(frozen=True)
class WeatherNormal:
    city: str
    month: int
    sample_days: int
    avg_temperature_2m_max_c: float
    std_temperature_2m_max_c: float
    avg_temperature_2m_min_c: float = 0.0
    std_temperature_2m_min_c: float = 0.0
    avg_precipitation_sum_mm: float = 0.0

    @property
    def avg_max_f(self) -> float:
        return _c_to_f(self.avg_temperature_2m_max_c)

    @property
    def std_max_f(self) -> float:
        return self.std_temperature_2m_max_c * 9 / 5

    @property
    def avg_min_f(self) -> float:
        return _c_to_f(self.avg_temperature_2m_min_c)

    @property
    def std_min_f(self) -> float:
        return self.std_temperature_2m_min_c * 9 / 5


class WeatherNormalStore:
    """Loads processed city/month weather normals for forecast base rates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._normals: dict[tuple[str, int], WeatherNormal] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    normal = WeatherNormal(
                        city=str(row["city"]),
                        month=int(row["month"]),
                        sample_days=int(row.get("sample_days") or 0),
                        avg_temperature_2m_max_c=float(row.get("avg_temperature_2m_max_c") or 0.0),
                        std_temperature_2m_max_c=float(row.get("std_temperature_2m_max_c") or 0.0),
                        avg_temperature_2m_min_c=float(row.get("avg_temperature_2m_min_c") or 0.0),
                        std_temperature_2m_min_c=float(row.get("std_temperature_2m_min_c") or 0.0),
                        avg_precipitation_sum_mm=float(row.get("avg_precipitation_sum_mm") or 0.0),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                self._normals[(normal.city.lower(), normal.month)] = normal

    def normal_for(self, city: str, target_date: str | dt.date) -> Optional[WeatherNormal]:
        if isinstance(target_date, str):
            try:
                month = dt.date.fromisoformat(target_date).month
            except ValueError:
                return None
        else:
            month = target_date.month
        return self._normals.get((city.lower(), month))

    def prompt_block(
        self,
        city: str,
        target_date: str | dt.date,
        consensus_temp_f: Optional[float],
    ) -> str:
        normal = self.normal_for(city, target_date)
        if normal is None:
            return ""

        anomaly = None
        sigma = None
        if consensus_temp_f is not None:
            anomaly = consensus_temp_f - normal.avg_max_f
            if normal.std_max_f > 0:
                sigma = anomaly / normal.std_max_f

        anomaly_line = "  Forecast max anomaly vs normal: N/A"
        if anomaly is not None:
            if sigma is not None:
                anomaly_line = (
                    f"  Forecast max anomaly vs normal: {anomaly:+.1f}F "
                    f"({sigma:+.2f} sigma)"
                )
            else:
                anomaly_line = f"  Forecast max anomaly vs normal: {anomaly:+.1f}F"

        return (
            "\nHistorical city/month baseline:\n"
            f"  Month: {normal.month} | sample days: {normal.sample_days}\n"
            f"  Normal max: {normal.avg_max_f:.1f}F "
            f"(std {normal.std_max_f:.1f}F)\n"
            f"  Normal min: {normal.avg_min_f:.1f}F "
            f"(std {normal.std_min_f:.1f}F)\n"
            f"  Avg daily precip: {normal.avg_precipitation_sum_mm:.2f} mm\n"
            f"{anomaly_line}\n"
            "Use this as a base-rate anchor, then adjust using current forecast models."
        )
