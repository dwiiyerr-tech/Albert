import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from remote_control.telegram_bot import (
    RemoteControlCommandHandler,
    RemoteControlPolicy,
    parse_command,
)
from trading.position_manager import PositionManager


class _FakeMemory:
    def overall_stats(self):
        return {
            "total_predictions": 12,
            "avg_brier_score": 0.22,
            "total_trades": 3,
            "win_rate": 0.5,
            "total_pnl_usd": 1.25,
            "total_lessons": 2,
        }

    def persona_score_report(self):
        return [
            {"name": "Alex", "predictions": 4, "avg_brier": 0.18, "weight": 1.2},
        ]


class _FakeAgent:
    def __init__(self, positions_file: str):
        self.dry_run = True
        self._demo = None
        self.positions = PositionManager(positions_file=positions_file)
        self.memory = _FakeMemory()
        self._cycle_state = {
            "cycle_num": 2,
            "current_city": "-",
            "last_signals": [
                SimpleNamespace(
                    city="Dallas",
                    direction="NO",
                    target_date="2026-05-17",
                    bucket_low=88.0,
                    bucket_high=89.0,
                    ev=0.25,
                    model_probability=0.90,
                    market_price=0.70,
                    recommended_usd=0.80,
                )
            ],
        }
        self.cycles_run = 0

    def run_cycle(self, days_ahead: int = 1):
        self.cycles_run += 1
        return self._cycle_state["last_signals"]


class RemoteControlTests(unittest.TestCase):
    def _handler(self, allowed_chat_ids: str = "123", allowed_commands: str = ""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        policy = RemoteControlPolicy.from_strings(
            allowed_chat_ids=allowed_chat_ids,
            allowed_commands=allowed_commands or "status,positions,signals,learning,dry_run_once",
            audit_log=str(Path(tmp.name) / "remote.log"),
        )
        agent = _FakeAgent(str(Path(tmp.name) / "positions.json"))
        return RemoteControlCommandHandler(agent, policy), agent

    def test_parse_telegram_command_alias(self):
        self.assertEqual(parse_command("/dry@AlbertBot now"), ("dry_run_once", "now"))

    def test_whoami_works_without_authorization(self):
        handler, _agent = self._handler(allowed_chat_ids="")
        response = handler.handle("/whoami", chat_id=999)

        self.assertIn("999", response)

    def test_unauthorized_chat_is_denied(self):
        handler, _agent = self._handler(allowed_chat_ids="123")
        response = handler.handle("/status", chat_id=999)

        self.assertIn("Akses ditolak", response)

    def test_command_allowlist_blocks_demo(self):
        handler, _agent = self._handler(allowed_commands="status")
        response = handler.handle("/demo_once", chat_id=123)

        self.assertIn("tidak diizinkan", response)

    def test_status_and_dry_run_once(self):
        handler, agent = self._handler()

        status = handler.handle("/status", chat_id=123)
        run = handler.handle("/dry_run_once", chat_id=123)

        self.assertIn("Albert Status", status)
        self.assertIn("Dry-run cycle selesai", run)
        self.assertEqual(agent.cycles_run, 1)


if __name__ == "__main__":
    unittest.main()
