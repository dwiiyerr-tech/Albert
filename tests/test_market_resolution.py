import unittest

from trading.market_resolution import PolymarketResolutionClient


class PolymarketResolutionClientTests(unittest.TestCase):
    def test_parse_resolved_no_market_from_gamma_prices(self) -> None:
        client = PolymarketResolutionClient()
        market = {
            "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0", "1"]',
        }

        self.assertEqual(client._payouts_from_market(market), (0.0, 1.0))

    def test_unclosed_market_does_not_resolve(self) -> None:
        client = PolymarketResolutionClient()
        market = {
            "closed": False,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.01", "0.99"]',
        }

        self.assertIsNone(client._payouts_from_market(market))

    def test_resolve_token_maps_condition_to_gamma_market(self) -> None:
        client = PolymarketResolutionClient()
        client._token_mapping = lambda token: {"condition_id": "cond-1"}
        client._gamma_market_by_condition = lambda condition: {
            "id": "market-1",
            "question": "Will it resolve yes?",
            "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["1", "0"]',
        }

        result = client.resolve_token("yes-token")

        self.assertIsNotNone(result)
        self.assertTrue(result.outcome_yes)
        self.assertEqual(result.condition_id, "cond-1")
        self.assertAlmostEqual(result.yes_payout, 1.0)
        self.assertAlmostEqual(result.no_payout, 0.0)

    def test_half_resolution_is_marked_unknown_but_payouts_are_available(self) -> None:
        client = PolymarketResolutionClient()
        client._token_mapping = lambda token: {"condition_id": "cond-1"}
        client._gamma_market_by_condition = lambda condition: {
            "id": "market-1",
            "question": "Unknown outcome?",
            "closed": True,
            "umaResolutionStatus": "resolved",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.5", "0.5"]',
        }

        result = client.resolve_token("yes-token")

        self.assertIsNotNone(result)
        self.assertIsNone(result.outcome_yes)
        self.assertAlmostEqual(result.yes_payout, 0.5)
        self.assertAlmostEqual(result.no_payout, 0.5)


if __name__ == "__main__":
    unittest.main()
