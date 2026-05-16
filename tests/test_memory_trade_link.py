import tempfile
import unittest
from pathlib import Path

from learning.memory import ExperienceMemory


class MemoryTradeLinkTests(unittest.TestCase):
    def test_attach_trade_to_latest_matching_prediction(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        memory = ExperienceMemory(memory_file=str(Path(tmp.name) / "memory.json"))

        memory.record_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=92,
            bucket_high=float("inf"),
            consensus_probability=0.30,
            confidence_level="medium",
            model_spread_f=1.0,
            agent_probabilities=[0.30],
            agent_estimates={"Analyst": 0.30},
            market_price=0.10,
            market_volume=1000,
            hours_to_resolution=20,
            market_id="yes-token",
            ecmwf_f=91,
            gfs_f=93,
            metar_f=None,
        )
        latest = memory.record_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=92,
            bucket_high=float("inf"),
            consensus_probability=0.35,
            confidence_level="medium",
            model_spread_f=1.0,
            agent_probabilities=[0.35],
            agent_estimates={"Analyst": 0.35},
            market_price=0.10,
            market_volume=1000,
            hours_to_resolution=20,
            market_id="yes-token",
            ecmwf_f=91,
            gfs_f=93,
            metar_f=None,
        )

        tagged = memory.attach_trade_to_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=92,
            bucket_high=float("inf"),
            market_id="yes-token",
            direction="YES",
            size_usd=1.25,
            ev=0.42,
        )

        self.assertEqual(tagged.id, latest.id)
        self.assertEqual(latest.trade_direction, "YES")
        self.assertAlmostEqual(latest.trade_size_usd, 1.25)
        self.assertAlmostEqual(latest.trade_ev, 0.42)

    def test_mark_trade_exit_updates_open_trade_record(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        memory = ExperienceMemory(memory_file=str(Path(tmp.name) / "memory.json"))
        rec = memory.record_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=92,
            bucket_high=float("inf"),
            consensus_probability=0.35,
            confidence_level="medium",
            model_spread_f=1.0,
            agent_probabilities=[0.35],
            agent_estimates={"Analyst": 0.35},
            market_price=0.10,
            market_volume=1000,
            hours_to_resolution=20,
            market_id="yes-token",
            ecmwf_f=91,
            gfs_f=93,
            metar_f=None,
        )
        memory.attach_trade_to_prediction(
            city="Dallas",
            target_date="2026-05-17",
            bucket_low=92,
            bucket_high=float("inf"),
            market_id="yes-token",
            direction="YES",
            size_usd=1.25,
            ev=0.42,
        )

        updated = memory.mark_trade_exit(
            market_id="yes-token",
            target_date="2026-05-17",
            pnl_usd=-0.25,
            reason="stop_loss",
        )

        self.assertEqual(updated.id, rec.id)
        self.assertAlmostEqual(rec.trade_pnl_usd, -0.25)
        self.assertEqual(rec.trade_close_reason, "stop_loss")
        self.assertIsNotNone(rec.trade_exit_ts)


if __name__ == "__main__":
    unittest.main()
