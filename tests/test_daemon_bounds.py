import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import MiroWeatherAgent


class _BoundedDaemonAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.memory = SimpleNamespace(
            lessons=[],
            resolved_records=lambda: [],
        )

    def run_cycle(self, days_ahead: int = 1):
        self.calls += 1
        return []


class DaemonBoundsTests(unittest.TestCase):
    def test_daemon_stops_after_max_cycles(self) -> None:
        agent = _BoundedDaemonAgent()

        with patch("main.time.sleep") as sleep:
            MiroWeatherAgent.daemon(agent, days_ahead=1, max_cycles=2)

        self.assertEqual(agent.calls, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
