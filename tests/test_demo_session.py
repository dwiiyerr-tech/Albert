import unittest

from demo.session import DemoSession
from llm_client import LLMClient


class DemoSessionTokenTrackingTests(unittest.TestCase):
    def test_repeated_demo_sessions_do_not_double_count_llm_calls(self) -> None:
        first = DemoSession()
        first.install_token_tracking()
        client = LLMClient(provider="mock", model="mock")
        client.text(system="system", messages=[{"role": "user", "content": "one"}])

        self.assertEqual(first.tokens.api_calls, 1)

        second = DemoSession()
        second.install_token_tracking()
        client.text(system="system", messages=[{"role": "user", "content": "two"}])

        self.assertEqual(first.tokens.api_calls, 1)
        self.assertEqual(second.tokens.api_calls, 1)


if __name__ == "__main__":
    unittest.main()
