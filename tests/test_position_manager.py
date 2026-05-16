import tempfile
import unittest
from pathlib import Path

from trading.position_manager import PositionManager


class PositionManagerSettlementTests(unittest.TestCase):
    def _manager(self) -> PositionManager:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return PositionManager(positions_file=str(Path(tmp.name) / "positions.json"))

    def test_no_position_wins_when_yes_outcome_is_false(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="NO",
            entry_price=0.25,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        closed = manager.resolve_position("market-1", outcome_yes=False)

        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.current_price, 1.0)
        self.assertAlmostEqual(closed.pnl_usd, 30.0)
        self.assertEqual(manager.open_positions, {})

    def test_no_position_loses_when_yes_outcome_is_true(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="NO",
            entry_price=0.25,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        closed = manager.resolve_position("market-1", outcome_yes=True)

        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.current_price, 0.0)
        self.assertAlmostEqual(closed.pnl_usd, -10.0)

    def test_no_stop_loss_uses_owned_no_token_price(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="NO",
            entry_price=0.40,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        reason = manager.update_price("market-1", 0.31)

        self.assertEqual(reason, "stop_loss")
        self.assertEqual(len(manager.closed_positions), 1)
        self.assertLess(manager.closed_positions[0].pnl_usd, 0)


if __name__ == "__main__":
    unittest.main()
