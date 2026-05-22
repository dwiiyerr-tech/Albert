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

    def test_stop_signal_can_leave_live_position_open_until_exit_order_succeeds(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.40,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        reason = manager.update_price("market-1", 0.31, close_on_trigger=False)

        self.assertEqual(reason, "stop_loss")
        self.assertIn("market-1", manager.open_positions)
        self.assertEqual(len(manager.closed_positions), 0)

    def test_half_payout_closes_at_half_value(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.25,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        closed = manager.resolve_position_payout("market-1", yes_payout=0.5, no_payout=0.5)

        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.current_price, 0.5)
        self.assertAlmostEqual(closed.pnl_usd, 10.0)

    def test_risk_snapshot_tracks_deployed_heat_and_daily_loss(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.50,
            size_usd=10.0,
            size_shares=20.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )
        manager.update_price("market-1", 0.25)

        snapshot = manager.risk_snapshot()

        self.assertEqual(snapshot["open_positions"], 0)
        self.assertEqual(snapshot["closed_today"], 1)
        self.assertEqual(snapshot["opened_today"], 1)
        self.assertGreater(snapshot["daily_loss_usd"], 0)
        self.assertGreater(snapshot["drawdown_usd"], 0)

    def test_live_position_tracks_filled_shares_for_pnl(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.25,
            size_usd=5.0,
            size_shares=20.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        manager.update_price("market-1", 0.50, close_on_trigger=False)

        self.assertAlmostEqual(manager.open_positions["market-1"].unrealized_pnl_usd, 5.0)

    def test_reduce_position_handles_partial_live_exit(self) -> None:
        manager = self._manager()
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.25,
            size_usd=10.0,
            size_shares=40.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        closed_fragment = manager.reduce_position("market-1", 0.50, 10.0, reason="stop_loss")

        self.assertIsNotNone(closed_fragment)
        self.assertIn("market-1", manager.open_positions)
        self.assertAlmostEqual(closed_fragment.pnl_usd, 2.5)
        self.assertAlmostEqual(manager.open_positions["market-1"].size_shares, 30.0)
        self.assertAlmostEqual(manager.open_positions["market-1"].size_usd, 7.5)

    def test_save_uses_atomic_replace_without_leaving_tmp_file(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "positions.json"
        manager = PositionManager(positions_file=str(path))
        manager.open_position(
            market_id="market-1",
            city="Miami",
            direction="YES",
            entry_price=0.25,
            size_usd=10.0,
            bucket_low=88,
            bucket_high=89,
            target_date="2026-05-17",
        )

        manager.save()

        self.assertTrue(path.exists())
        self.assertFalse(Path(f"{path}.tmp").exists())
        reloaded = PositionManager(positions_file=str(path))
        self.assertIn("market-1", reloaded.open_positions)


if __name__ == "__main__":
    unittest.main()
