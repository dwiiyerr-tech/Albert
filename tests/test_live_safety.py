import datetime
import unittest

from main import (
    LIVE_TRADING_CONFIRM_PHRASE,
    MiroWeatherAgent,
    live_trading_block_reason,
)


class LiveSafetyTests(unittest.TestCase):
    def test_live_is_blocked_by_default(self) -> None:
        reason = live_trading_block_reason(
            today=datetime.date(2026, 5, 18),
            enabled=False,
            confirmation="",
            demo_only_until="",
        )

        self.assertIn("LIVE_TRADING_ENABLED", reason)

    def test_live_requires_exact_confirmation_phrase(self) -> None:
        reason = live_trading_block_reason(
            today=datetime.date(2026, 5, 18),
            enabled=True,
            confirmation="yes",
            demo_only_until="",
        )

        self.assertIn("LIVE_TRADING_CONFIRM", reason)

    def test_demo_only_until_blocks_even_when_enabled(self) -> None:
        reason = live_trading_block_reason(
            today=datetime.date(2026, 5, 18),
            enabled=True,
            confirmation=LIVE_TRADING_CONFIRM_PHRASE,
            demo_only_until="2026-06-08",
        )

        self.assertIn("demo-only lock", reason)

    def test_live_can_only_pass_after_all_safety_settings_are_valid(self) -> None:
        reason = live_trading_block_reason(
            today=datetime.date(2026, 6, 9),
            enabled=True,
            confirmation=LIVE_TRADING_CONFIRM_PHRASE,
            demo_only_until="2026-06-08",
        )

        self.assertEqual(reason, "")

    def test_agent_constructor_blocks_programmatic_live_mode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Live trading is blocked"):
            MiroWeatherAgent(dry_run=False)


if __name__ == "__main__":
    unittest.main()
