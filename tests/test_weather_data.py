import unittest
from unittest.mock import patch

import weather_data


class WeatherRetryTests(unittest.TestCase):
    def test_retry_get_does_not_sleep_after_final_failure(self) -> None:
        calls = []
        sleeps = []

        def fake_get(url, params, timeout):
            calls.append((url, params, timeout))
            raise RuntimeError("network down")

        with patch.object(weather_data.requests, "get", side_effect=fake_get), \
             patch.object(weather_data.time, "sleep", side_effect=lambda seconds: sleeps.append(seconds)), \
             patch.object(weather_data.logger, "warning"):
            result = weather_data._retry_get(
                "https://example.invalid",
                {},
                retries=3,
                backoff=(1, 2, 4),
                timeout=1,
            )

        self.assertIsNone(result)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1, 2])


if __name__ == "__main__":
    unittest.main()
