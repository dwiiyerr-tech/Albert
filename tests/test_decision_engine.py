import unittest

from learning.feature_store import MarketFeatures
from trading.decision_engine import DecisionEngine
from trading.ev_calculator import TradeSignal


def _signal(direction: str = "YES") -> TradeSignal:
    return TradeSignal(
        city="Dallas",
        target_date="2026-05-17",
        bucket_low=80,
        bucket_high=90,
        direction=direction,
        probability=0.65,
        market_price=0.50,
        ev=0.30,
        kelly_fraction=0.10,
        recommended_usd=5.0,
        confidence_level="high",
        hours_to_resolution=12,
        volume=1000,
        market_id="yes-token",
        no_token_id="no-token",
        condition_id="cond-1",
        probability_edge=0.15,
        orderbook_depth_usd=50,
        slippage=0.0,
    )


class DecisionEngineTests(unittest.TestCase):
    def test_low_feature_quality_allows_paper_but_blocks_live_readiness(self) -> None:
        features = MarketFeatures(
            token_id="yes-token",
            condition_id="cond-1",
            data_quality_score=0.0,
            missing_sources=["price_history", "trades", "holders", "open_interest"],
        )

        decision = DecisionEngine().evaluate(_signal(), features, live_mode=True)

        self.assertEqual(decision.action, "PAPER_TRADE")
        self.assertFalse(decision.live_ready)
        self.assertLess(decision.risk_multiplier, 1.0)

    def test_high_quality_features_can_be_live_ready(self) -> None:
        features = MarketFeatures(
            token_id="yes-token",
            condition_id="cond-1",
            data_quality_score=0.90,
            latest_price=0.51,
            trade_count_24h=12,
            holder_count=8,
            top_holder_concentration=0.20,
            open_interest=1500,
        )

        decision = DecisionEngine().evaluate(_signal(), features, live_mode=True)

        self.assertEqual(decision.action, "LIVE_READY")
        self.assertTrue(decision.live_ready)
        self.assertEqual(decision.risk_multiplier, 1.0)

    def test_concentrated_holder_market_moves_to_watch(self) -> None:
        features = MarketFeatures(
            token_id="yes-token",
            condition_id="cond-1",
            data_quality_score=0.80,
            top_holder_concentration=0.90,
        )

        decision = DecisionEngine().evaluate(_signal(), features, live_mode=False)

        self.assertEqual(decision.action, "WATCH")
        self.assertFalse(decision.should_execute_paper)


if __name__ == "__main__":
    unittest.main()
