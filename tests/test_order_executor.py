import unittest

from trading.order_executor import PolymarketOrderExecutor


class _FakeClient:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_order(self, order_id):
        return self.snapshot


class OrderExecutorSizingTests(unittest.TestCase):
    def _executor(self, snapshot=None):
        executor = PolymarketOrderExecutor.__new__(PolymarketOrderExecutor)
        executor._library_available = True
        executor._private_key = "test-key"
        executor._client = _FakeClient(snapshot or {})
        return executor

    def test_shares_for_usd_converts_notional_to_token_size(self) -> None:
        executor = self._executor()

        self.assertAlmostEqual(executor._shares_for_usd(10.0, 0.25), 40.0)

    def test_execution_parser_uses_confirmed_fill_snapshot(self) -> None:
        executor = self._executor({
            "status": "MATCHED",
            "filled_size": "40",
            "average_price": "0.25",
        })

        execution = executor._execution_from_response(
            response={"orderID": "order-1"},
            order_id="order-1",
            token_id="token-1",
            side="BUY",
            limit_price=0.25,
            requested_shares=40.0,
            requested_usd=10.0,
        )

        self.assertTrue(execution.has_fill)
        self.assertTrue(execution.is_fully_filled)
        self.assertEqual(execution.status, "matched")
        self.assertAlmostEqual(execution.filled_usd, 10.0)

    def test_execution_parser_detects_unfilled_order(self) -> None:
        executor = self._executor({"status": "LIVE", "size": "40", "remaining_size": "40"})

        execution = executor._execution_from_response(
            response={"orderID": "order-1"},
            order_id="order-1",
            token_id="token-1",
            side="BUY",
            limit_price=0.25,
            requested_shares=40.0,
            requested_usd=10.0,
        )

        self.assertFalse(execution.has_fill)
        self.assertEqual(execution.status, "live")


if __name__ == "__main__":
    unittest.main()
