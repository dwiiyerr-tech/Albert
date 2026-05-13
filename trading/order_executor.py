"""
Polymarket CLOB Order Executor.

Submits EIP-712-signed limit orders to the Polymarket Central Limit Order Book
via py-clob-client. Handles L1 (private key) and L2 (derived API key) auth.

Graceful degradation: if py-clob-client is not installed OR
POLYMARKET_PRIVATE_KEY is empty, is_configured() returns False and
all methods are safe no-ops (return None / False / 0.0).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trading.ev_calculator import TradeSignal

logger = logging.getLogger(__name__)

_POLYGON_CHAIN_ID = 137   # Polygon mainnet where Polymarket lives


class PolymarketOrderExecutor:
    """
    Thin wrapper around py-clob-client for order signing and submission.

    Trade flow:
      YES signal → BUY YES token  (token_id = signal.market_id)
      NO  signal → BUY NO  token  (token_id = signal.no_token_id)

    All public methods catch every exception and return a sentinel value so a
    broken order path never crashes the agent cycle.
    """

    def __init__(
        self,
        private_key: str,
        api_key: str = "",
        proxy_address: str = "",
        chain_id: int = _POLYGON_CHAIN_ID,
        host: str = "https://clob.polymarket.com",
    ) -> None:
        self._private_key = private_key.strip()
        self._proxy_address = proxy_address.strip()
        self._chain_id = chain_id
        self._host = host
        self._client = None

        self._library_available = self._try_import()
        if self._library_available and self._private_key:
            self._init_client()

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _try_import(self) -> bool:
        try:
            import py_clob_client  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "py-clob-client not installed — live order execution disabled. "
                "Install with: pip install 'py-clob-client>=0.6.0'"
            )
            return False

    def _init_client(self) -> None:
        """Construct ClobClient and derive L2 API credentials from private key."""
        try:
            from py_clob_client.client import ClobClient

            client = ClobClient(
                host=self._host,
                chain_id=self._chain_id,
                key=self._private_key,
                signature_type=0,                      # EOA (non-proxy wallet)
                funder=self._proxy_address or None,
            )

            # Derive L2 API key — deterministic from the private key.
            # Try derive_api_key() (>=0.6) then create_api_key() (0.5.x).
            try:
                creds = client.derive_api_key()
            except AttributeError:
                try:
                    creds = client.create_api_key()
                except Exception as exc:
                    logger.warning("Could not derive API credentials: %s", exc)
                    creds = None

            if creds:
                client.set_api_creds(creds)

            self._client = client
            logger.info("PolymarketOrderExecutor ready (chain_id=%d)", self._chain_id)

        except Exception as exc:
            logger.warning("PolymarketOrderExecutor init failed: %s", exc)
            self._client = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """True only when library is installed, key is set, and client initialised."""
        return (
            self._library_available
            and bool(self._private_key)
            and self._client is not None
        )

    def place_order(self, signal: "TradeSignal") -> Optional[str]:
        """
        Sign and submit a GTC limit order.
        Returns the CLOB order_id on success, None on any failure.

        YES → BUY YES token at signal.market_price
        NO  → BUY NO  token at signal.market_price (already the NO token price)
        """
        if not self.is_configured():
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.constants import BUY

            if signal.direction == "YES":
                token_id = signal.market_id
                price = signal.market_price
            else:
                token_id = signal.no_token_id
                price = signal.market_price   # ev_calculator already flipped this to NO price

            if not token_id:
                logger.warning(
                    "No token_id for %s %s — cannot place order",
                    signal.direction, signal.city,
                )
                return None

            price = round(max(0.001, min(0.999, price)), 4)
            size = round(signal.recommended_usd, 2)

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,
            )
            signed_order = self._client.create_order(order_args)
            response = self._client.post_order(signed_order, OrderType.GTC)

            order_id: Optional[str] = None
            if isinstance(response, dict):
                order_id = response.get("orderID") or response.get("order_id")
            elif hasattr(response, "order_id"):
                order_id = response.order_id

            if order_id:
                logger.info(
                    "LIVE ORDER: %s %s @ %.4f $%.2f → order_id=%s",
                    signal.direction, signal.city, price, size, order_id,
                )
            else:
                logger.warning("Order posted but no order_id in response: %s", response)

            return order_id

        except Exception as exc:
            logger.warning(
                "place_order failed for %s %s: %s",
                signal.direction, signal.city, exc,
            )
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if accepted by CLOB."""
        if not self.is_configured() or not order_id:
            return False
        try:
            self._client.cancel(order_id)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.warning("cancel_order %s failed: %s", order_id, exc)
            return False

    def get_balance(self) -> float:
        """Return available USDC balance in dollars. Returns 0.0 on failure."""
        if not self.is_configured():
            return 0.0
        try:
            raw = self._client.get_balance_allowance()
            if isinstance(raw, dict):
                raw_balance = float(raw.get("balance", 0))
            else:
                raw_balance = float(raw)
            return raw_balance / 1_000_000   # USDC has 6 decimals
        except Exception as exc:
            logger.warning("get_balance failed: %s", exc)
            return 0.0
