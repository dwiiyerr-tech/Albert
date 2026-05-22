import unittest
import datetime

from trading.market_scanner import MarketScanner


class MarketScannerMarkToMarketTests(unittest.TestCase):
    def test_exit_price_uses_best_bid(self) -> None:
        scanner = MarketScanner()
        scanner._book_prices = lambda token_id: {"best_bid": 0.37}

        self.assertEqual(scanner.get_token_exit_price("token-1"), 0.37)

    def test_exit_price_returns_none_without_book(self) -> None:
        scanner = MarketScanner()
        scanner._book_prices = lambda token_id: None

        self.assertIsNone(scanner.get_token_exit_price("token-1"))

    def test_market_entry_accepts_end_date_iso_alias(self) -> None:
        scanner = MarketScanner()

        def fake_book(token_id):
            return {
                "best_ask": 0.40 if token_id == "yes-token" else 0.60,
                "effective_ask": 0.41 if token_id == "yes-token" else 0.61,
                "best_bid": 0.39 if token_id == "yes-token" else 0.59,
                "mid": 0.40 if token_id == "yes-token" else 0.60,
                "spread": 0.02,
                "ask_depth_usd": 100.0,
                "slippage": 0.01,
            }

        scanner._book_prices = fake_book
        end_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
        market = {
            "question": "Will Dallas be 80-90F?",
            "clobTokenIds": '["yes-token","no-token"]',
            "conditionId": "cond-1",
            "endDateIso": end_at.isoformat(),
            "volume": "1000",
            "active": True,
            "closed": False,
        }

        entry = scanner._fetch_market_entry(market)

        self.assertIsNotNone(entry)
        self.assertGreater(entry["hours_to_resolution"], 23)
        self.assertEqual(entry["condition_id"], "cond-1")


if __name__ == "__main__":
    unittest.main()
