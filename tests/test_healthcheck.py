import unittest
from unittest.mock import patch

import healthcheck


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._payload = payload or {"ok": True, "result": {"username": "AlbertTestBot"}}

    def json(self) -> dict:
        return self._payload


class HealthcheckTests(unittest.TestCase):
    def test_local_healthcheck_passes_without_network(self) -> None:
        results = healthcheck.run_checks(skip_network=True)

        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["config"].status, "ok")
        self.assertEqual(by_name["state-files"].status, "ok")
        self.assertEqual(by_name["llm"].status, "ok")
        self.assertEqual(by_name["network"].status, "warn")

    def test_main_returns_zero_when_only_network_is_skipped(self) -> None:
        self.assertEqual(healthcheck.main(["--skip-network"]), 0)

    def test_strict_mode_fails_on_warnings(self) -> None:
        self.assertEqual(healthcheck.main(["--skip-network", "--strict"]), 1)

    def test_telegram_check_warns_when_disabled_and_unconfigured(self) -> None:
        with (
            patch.object(healthcheck.config, "REMOTE_CONTROL_PROVIDER", "telegram"),
            patch.object(healthcheck.config, "REMOTE_CONTROL_ENABLED", False),
            patch.object(healthcheck.config, "TELEGRAM_BOT_TOKEN", ""),
        ):
            result = healthcheck.check_telegram()

        self.assertEqual(result.status, "warn")
        self.assertIn("TELEGRAM_BOT_TOKEN", result.detail)

    def test_telegram_check_does_not_echo_token(self) -> None:
        token = "123456:secret-token"
        with (
            patch.object(healthcheck.config, "REMOTE_CONTROL_PROVIDER", "telegram"),
            patch.object(healthcheck.config, "TELEGRAM_BOT_TOKEN", token),
            patch.object(healthcheck.config, "REMOTE_ALLOWED_CHAT_IDS", "1,2"),
            patch.object(healthcheck.config, "REMOTE_NOTIFICATION_CHAT_IDS", "3"),
            patch.object(healthcheck.requests, "get", return_value=_FakeResponse()) as get,
        ):
            result = healthcheck.check_telegram()

        self.assertEqual(result.status, "ok")
        self.assertIn(token, get.call_args.args[0])
        self.assertNotIn(token, result.detail)
        self.assertIn("allowed_chats=2", result.detail)

    def test_telegram_check_fails_on_invalid_token(self) -> None:
        token = "123456:bad-token"
        response = _FakeResponse(
            status_code=404,
            payload={"ok": False, "description": "Not Found"},
        )
        with (
            patch.object(healthcheck.config, "REMOTE_CONTROL_PROVIDER", "telegram"),
            patch.object(healthcheck.config, "TELEGRAM_BOT_TOKEN", token),
            patch.object(healthcheck.requests, "get", return_value=response),
        ):
            result = healthcheck.check_telegram()

        self.assertEqual(result.status, "fail")
        self.assertIn("Not Found", result.detail)
        self.assertNotIn(token, result.detail)


if __name__ == "__main__":
    unittest.main()
