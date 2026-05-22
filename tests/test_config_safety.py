import os
import subprocess
import sys
import unittest


class ConfigSafetyTests(unittest.TestCase):
    def test_invalid_numeric_env_values_do_not_break_import(self) -> None:
        env = os.environ.copy()
        env.update({
            "LLM_PROVIDER": "mock",
            "REMOTE_POLL_INTERVAL_SECONDS": "soon",
            "MAX_DAILY_TRADES": "many",
            "MIN_EV": "not-a-number",
            "UPDATE_INTERVAL_SECONDS": "later",
        })

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import config; "
                    "print(config.REMOTE_POLL_INTERVAL_SECONDS, "
                    "config.MAX_DAILY_TRADES, config.MIN_EV, "
                    "config.UPDATE_INTERVAL_SECONDS)"
                ),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("2.0 20 0.1 3600", completed.stdout)
        self.assertIn("invalid numeric config REMOTE_POLL_INTERVAL_SECONDS", completed.stderr)
        self.assertIn("invalid integer config MAX_DAILY_TRADES", completed.stderr)


if __name__ == "__main__":
    unittest.main()
