import unittest
import tempfile
from pathlib import Path

from learning.decision_scorecard import build_decision_scorecard
from learning.memory import ExperienceMemory


class DecisionScorecardTests(unittest.TestCase):
    def test_scores_resolved_trades_and_confidence_groups(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        memory = ExperienceMemory(memory_file=str(Path(tmp.name) / "memory.json"))

        yes_win = memory.record_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=80,
            bucket_high=90,
            consensus_probability=0.70,
            confidence_level="high",
            model_spread_f=1.0,
            agent_probabilities=[0.70],
            market_price=0.50,
            market_volume=1000,
            hours_to_resolution=12,
            market_id="market-1",
            ecmwf_f=85,
            gfs_f=84,
            metar_f=None,
        )
        yes_win.trade_direction = "YES"
        yes_win.trade_size_usd = 10
        yes_win.trade_ev = 0.20
        memory.resolve_prediction(yes_win.id, 86, True)

        no_win = memory.record_prediction(
            city="Austin",
            target_date="2026-05-17",
            bucket_low=80,
            bucket_high=90,
            consensus_probability=0.35,
            confidence_level="medium",
            model_spread_f=2.0,
            agent_probabilities=[0.35],
            market_price=0.40,
            market_volume=1000,
            hours_to_resolution=12,
            market_id="market-2",
            ecmwf_f=75,
            gfs_f=76,
            metar_f=None,
        )
        no_win.trade_direction = "NO"
        no_win.trade_size_usd = 12
        no_win.trade_ev = 0.18
        memory.resolve_prediction(no_win.id, 75, False)

        scorecard = build_decision_scorecard(memory)

        self.assertEqual(scorecard.resolved_predictions, 2)
        self.assertEqual(scorecard.traded_predictions, 2)
        self.assertEqual(scorecard.win_rate, 1.0)
        self.assertGreater(scorecard.total_pnl_usd, 0)
        self.assertIn("high", scorecard.by_confidence)
        self.assertIn("medium", scorecard.by_confidence)


if __name__ == "__main__":
    unittest.main()
