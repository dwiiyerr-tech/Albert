import json
import tempfile
import unittest
import datetime as dt
from pathlib import Path

from learning.feature_store import MarketFeatureStore
from trading.ev_calculator import TradeSignal


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class MarketFeatureStoreTests(unittest.TestCase):
    def test_builds_market_features_from_local_polymarket_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            two_hours_ago = (now - dt.timedelta(hours=2)).isoformat()
            one_day_ago = (now - dt.timedelta(hours=24)).isoformat()
            thirty_minutes_ago = (now - dt.timedelta(minutes=30)).isoformat()
            twenty_minutes_ago = (now - dt.timedelta(minutes=20)).isoformat()
            one_hour_ago = (now - dt.timedelta(hours=1)).isoformat()
            _write_jsonl(raw / "price_history_sample.jsonl", [
                {"token_id": "yes-token", "timestamp_iso": two_hours_ago, "price": 0.40},
                {"token_id": "yes-token", "timestamp_iso": now.isoformat(), "price": 0.50},
            ])
            _write_jsonl(raw / "data_api_trades.jsonl", [
                {"condition_id": "cond-1", "timestamp_iso": thirty_minutes_ago, "side": "BUY", "size": 100, "price": 0.50},
                {"condition_id": "cond-1", "timestamp_iso": twenty_minutes_ago, "side": "SELL", "size": 25, "price": 0.48},
            ])
            _write_jsonl(raw / "holders.jsonl", [
                {"token_id": "yes-token", "proxy_wallet": "0x1", "amount": 60},
                {"token_id": "yes-token", "proxy_wallet": "0x2", "amount": 40},
            ])
            _write_jsonl(raw / "open_interest.jsonl", [
                {"condition_id": "cond-1", "collected_at": one_day_ago, "open_interest": 1000},
                {"condition_id": "cond-1", "collected_at": now.isoformat(), "open_interest": 1200},
            ])
            _write_jsonl(raw / "user_activity.jsonl", [
                {"condition_id": "cond-1", "timestamp_iso": one_hour_ago, "type": "TRADE"},
            ])

            store = MarketFeatureStore(raw)
            signal = TradeSignal(
                city="Dallas",
                target_date="2026-05-17",
                bucket_low=80,
                bucket_high=90,
                direction="YES",
                probability=0.65,
                market_price=0.50,
                ev=0.30,
                kelly_fraction=0.10,
                recommended_usd=5.0,
                confidence_level="high",
                hours_to_resolution=12,
                volume=1000,
                market_id="yes-token",
                condition_id="cond-1",
                probability_edge=0.15,
                orderbook_depth_usd=50,
            )

            features = store.features_for_signal(signal)

            self.assertAlmostEqual(features.latest_price, 0.50)
            self.assertAlmostEqual(features.price_momentum_1h, 0.25)
            self.assertEqual(features.trade_count_1h, 2)
            self.assertGreater(features.trade_imbalance_1h, 0)
            self.assertEqual(features.holder_count, 2)
            self.assertAlmostEqual(features.top_holder_concentration, 0.60)
            self.assertAlmostEqual(features.open_interest, 1200)
            self.assertAlmostEqual(features.open_interest_change_24h, 0.20)
            self.assertGreaterEqual(features.data_quality_score, 0.90)


if __name__ == "__main__":
    unittest.main()
