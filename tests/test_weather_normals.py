import datetime as dt
import tempfile
import unittest
from pathlib import Path

from data_ingestion.storage import write_jsonl
from learning.weather_normals import WeatherNormalStore
from simulation.agents import WeatherSimulation
from weather_data import CityForecast, TemperatureReading


class WeatherNormalsPromptTests(unittest.TestCase):
    def _store(self) -> WeatherNormalStore:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "weather_city_month_normals.jsonl"
        write_jsonl(path, [{
            "city": "Dallas",
            "month": 5,
            "sample_days": 217,
            "avg_temperature_2m_max_c": 30.0,
            "std_temperature_2m_max_c": 3.0,
            "avg_temperature_2m_min_c": 19.0,
            "std_temperature_2m_min_c": 2.0,
            "avg_precipitation_sum_mm": 2.5,
        }])
        return WeatherNormalStore(path)

    def test_prompt_block_contains_base_rate_and_anomaly(self) -> None:
        block = self._store().prompt_block("Dallas", "2026-05-17", 95.0)

        self.assertIn("Historical city/month baseline", block)
        self.assertIn("Normal max: 86.0F", block)
        self.assertIn("Forecast max anomaly vs normal: +9.0F", block)

    def test_simulation_prompt_includes_weather_normals(self) -> None:
        sim = WeatherSimulation.__new__(WeatherSimulation)
        sim._weather_normals = self._store()
        forecast = CityForecast(city="Dallas", lat=32.7767, lon=-96.7970)
        forecast.ecmwf = TemperatureReading(
            source="ECMWF",
            city="Dallas",
            date=dt.date(2026, 5, 17),
            temp_c=35.0,
            confidence=1.0,
        )

        prompt = sim._morgan_system_prompt(forecast, 90, 100, "2026-05-17")

        self.assertIn("Historical city/month baseline", prompt)
        self.assertIn("Use this as a base-rate anchor", prompt)


if __name__ == "__main__":
    unittest.main()
