import unittest

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


if __name__ == "__main__":
    unittest.main()
