import unittest
import datetime as dt

from data_ingestion.collectors import (
    DataCollector,
    flatten_polymarket_events,
    rows_from_binance_klines,
    rows_from_openmeteo_daily,
)


class DataIngestionParsingTests(unittest.TestCase):
    def test_flatten_polymarket_public_search(self) -> None:
        payload = {
            "events": [{
                "id": "event-1",
                "title": "Weather event",
                "slug": "weather-event",
                "markets": [{
                    "id": "market-1",
                    "conditionId": "cond-1",
                    "question": "Will NYC be above 70F?",
                    "active": True,
                    "closed": False,
                    "clobTokenIds": '["yes-token","no-token"]',
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["0.42","0.58"]',
                    "volume": "123.45",
                }],
            }]
        }

        rows = flatten_polymarket_events(payload, collected_at="2026-05-16T00:00:00+00:00")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_id"], "event-1")
        self.assertEqual(rows[0]["yes_token_id"], "yes-token")
        self.assertEqual(rows[0]["no_token_id"], "no-token")
        self.assertEqual(rows[0]["outcome_prices"], ["0.42", "0.58"])
        self.assertAlmostEqual(rows[0]["volume"], 123.45)

    def test_openmeteo_daily_rows(self) -> None:
        data = {
            "daily": {
                "time": ["2026-05-16"],
                "temperature_2m_max": [31.0],
                "temperature_2m_min": [20.0],
                "precipitation_sum": [1.2],
                "wind_speed_10m_max": [14.5],
            }
        }
        city = {"name": "Dallas", "lat": 32.7767, "lon": -96.7970}

        rows = rows_from_openmeteo_daily(data, city, "gfs_seamless")

        self.assertEqual(rows[0]["city"], "Dallas")
        self.assertEqual(rows[0]["model"], "gfs_seamless")
        self.assertEqual(rows[0]["temperature_2m_max_c"], 31.0)
        self.assertIsNone(rows[0]["temperature_2m_mean_c"])

    def test_binance_kline_rows(self) -> None:
        rows = rows_from_binance_klines(
            [[1609459200000, "1", "2", "0.5", "1.5", "10", 1609545599999, "15", 20, "4", "6", "0"]],
            "BTCUSDT",
            "1d",
        )

        self.assertEqual(rows[0]["symbol"], "BTCUSDT")
        self.assertEqual(rows[0]["interval"], "1d")
        self.assertEqual(rows[0]["open_time"], "2021-01-01T00:00:00+00:00")
        self.assertAlmostEqual(rows[0]["close"], 1.5)
        self.assertEqual(rows[0]["trade_count"], 20)

    def test_open_interest_paginates(self) -> None:
        calls = []

        def fake_get(url, params):
            calls.append(params["startTime"])
            if len(calls) == 1:
                return [{
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "10",
                    "sumOpenInterestValue": "100",
                    "timestamp": 1609459200000,
                }]
            if len(calls) == 2:
                return [{
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "11",
                    "sumOpenInterestValue": "110",
                    "timestamp": 1609545600000,
                }]
            return []

        collector = DataCollector(get_json=fake_get, sleep_seconds=0.0)
        rows = collector.collect_binance_open_interest(
            "BTCUSDT",
            "1d",
            dt.date(2021, 1, 1),
            dt.date(2021, 1, 3),
        )

        self.assertEqual(len(rows), 2)
        self.assertGreater(calls[1], calls[0])
        self.assertEqual(rows[1]["timestamp"], "2021-01-02T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
