import unittest
import datetime as dt
import tempfile
from pathlib import Path

from data_ingestion.collectors import (
    DataCollector,
    flatten_polymarket_events,
    rows_from_binance_klines,
    rows_from_openmeteo_daily,
    rows_from_polymarket_holders,
    rows_from_polymarket_open_interest,
    rows_from_polymarket_price_history,
    rows_from_polymarket_trades,
    select_polymarket_targets,
)
from data_ingestion.features import build_btc_risk_regimes, build_weather_monthly_normals
from data_ingestion.storage import write_jsonl
from learning.risk_regime import RiskRegimeStore


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

    def test_polymarket_price_history_rows(self) -> None:
        rows = rows_from_polymarket_price_history(
            {"history": [{"t": 1609459200, "p": 0.42}]},
            "token-1",
            "1d",
            collected_at="2026-05-16T00:00:00+00:00",
        )

        self.assertEqual(rows[0]["token_id"], "token-1")
        self.assertEqual(rows[0]["timestamp_iso"], "2021-01-01T00:00:00+00:00")
        self.assertAlmostEqual(rows[0]["price"], 0.42)

    def test_polymarket_flow_rows(self) -> None:
        trades = rows_from_polymarket_trades([{
            "proxyWallet": "0xabc",
            "conditionId": "0xcond",
            "side": "BUY",
            "size": "10",
            "price": "0.51",
            "timestamp": 1609459200,
            "transactionHash": "0xtx",
        }])
        holders = rows_from_polymarket_holders([{
            "token": "token-1",
            "holders": [{"proxyWallet": "0xabc", "amount": "25", "outcomeIndex": 1}],
        }])
        oi = rows_from_polymarket_open_interest([{"market": "0xcond", "value": "1250.5"}])

        self.assertEqual(trades[0]["condition_id"], "0xcond")
        self.assertAlmostEqual(trades[0]["price"], 0.51)
        self.assertEqual(holders[0]["token_id"], "token-1")
        self.assertAlmostEqual(holders[0]["amount"], 25.0)
        self.assertEqual(oi[0]["condition_id"], "0xcond")
        self.assertAlmostEqual(oi[0]["open_interest"], 1250.5)

    def test_polymarket_collector_uses_official_data_endpoints(self) -> None:
        calls = []

        def fake_get(url, params):
            calls.append((url, params))
            if url.endswith("/trades"):
                return [{"conditionId": "0xcond", "price": "0.4"}]
            if url.endswith("/holders"):
                return [{"token": "token-1", "holders": [{"proxyWallet": "0xabc", "amount": "3"}]}]
            if url.endswith("/oi"):
                return [{"market": "0xcond", "value": "99"}]
            if url.endswith("/activity"):
                return [{"proxyWallet": "0xabc", "type": "TRADE", "timestamp": 1609459200}]
            return {}

        collector = DataCollector(get_json=fake_get, sleep_seconds=0.0)
        self.assertEqual(len(collector.collect_polymarket_trades(["0xcond"], limit=5)), 1)
        self.assertEqual(len(collector.collect_polymarket_holders(["0xcond"], limit=100)), 1)
        self.assertEqual(len(collector.collect_polymarket_open_interest(["0xcond"])), 1)
        self.assertEqual(len(collector.collect_polymarket_user_activity("0xabc", ["0xcond"])), 1)

        self.assertEqual(calls[0][0], "https://data-api.polymarket.com/trades")
        self.assertEqual(calls[0][1]["market"], "0xcond")
        self.assertEqual(calls[1][1]["limit"], 20)
        self.assertEqual(calls[3][1]["user"], "0xabc")

    def test_select_polymarket_targets_prefers_liquid_markets(self) -> None:
        rows = [
            {
                "condition_id": "cond-small",
                "yes_token_id": "yes-small",
                "no_token_id": "no-small",
                "volume": 10,
                "liquidity": 5,
            },
            {
                "condition_id": "cond-big",
                "yes_token_id": "yes-big",
                "no_token_id": "no-big",
                "volume": 100,
                "liquidity": 20,
            },
        ]

        targets = select_polymarket_targets(rows, max_markets=1)

        self.assertEqual(targets["condition_ids"], ["cond-big"])
        self.assertEqual(targets["token_ids"], ["yes-big", "no-big"])

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

    def test_btc_regime_builder_labels_crash_and_throttles_risk(self) -> None:
        candles = []
        for i in range(35):
            close = 100.0 + i
            if i == 34:
                close = 80.0
            candles.append({
                "symbol": "BTCUSDT",
                "open_time": (dt.date(2021, 1, 1) + dt.timedelta(days=i)).isoformat(),
                "close": close,
            })

        rows = build_btc_risk_regimes(candles)

        self.assertEqual(rows[-1]["regime"], "crash")
        self.assertLess(rows[-1]["risk_multiplier"], 1.0)

    def test_btc_regime_builder_separates_deep_drawdown_from_crash(self) -> None:
        candles = []
        for i in range(260):
            close = 100.0 + i
            if i >= 220:
                close = 180.0 + ((i - 220) * 0.2)
            candles.append({
                "symbol": "BTCUSDT",
                "open_time": (dt.date(2021, 1, 1) + dt.timedelta(days=i)).isoformat(),
                "close": close,
            })

        rows = build_btc_risk_regimes(candles)

        self.assertIn(rows[-1]["regime"], {"deep_drawdown", "recovery"})
        self.assertNotEqual(rows[-1]["regime"], "crash")

    def test_weather_normals_group_by_city_month(self) -> None:
        rows = build_weather_monthly_normals([
            {"city": "Dallas", "date": "2021-01-01", "temperature_2m_max_c": 10, "temperature_2m_min_c": 1},
            {"city": "Dallas", "date": "2021-01-02", "temperature_2m_max_c": 12, "temperature_2m_min_c": 3},
            {"city": "Dallas", "date": "2021-02-01", "temperature_2m_max_c": 20, "temperature_2m_min_c": 8},
        ])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["city"], "Dallas")
        self.assertEqual(rows[0]["month"], 1)
        self.assertAlmostEqual(rows[0]["avg_temperature_2m_max_c"], 11.0)

    def test_risk_regime_store_reads_latest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regimes.jsonl"
            write_jsonl(path, [
                {"date": "2021-01-01", "regime": "neutral", "risk_multiplier": 1.0},
                {"date": "2021-01-02", "regime": "crash", "risk_multiplier": 0.25},
            ])

            latest = RiskRegimeStore(path).latest()

        self.assertIsNotNone(latest)
        self.assertEqual(latest.regime, "crash")
        self.assertEqual(latest.risk_multiplier, 0.25)


if __name__ == "__main__":
    unittest.main()
