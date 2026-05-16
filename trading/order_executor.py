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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trading.ev_calculator import TradeSignal
    from trading.position_manager import Position

from config import IOC_URGENCY_HOURS, ORDER_RETRY_MAX

logger = logging.getLogger(__name__)

_POLYGON_CHAIN_ID = 137   # Polygon mainnet where Polymarket lives


@dataclass
class OrderExecution:
    order_id: str
    token_id: str
    side: str
    limit_price: float
    requested_shares: float
    requested_usd: float
    filled_shares: float = 0.0
    filled_usd: float = 0.0
    average_price: float = 0.0
    status: str = "submitted"

    @property
    def has_fill(self) -> bool:
        return self.filled_shares > 0 and self.filled_usd > 0

    @property
    def is_fully_filled(self) -> bool:
        return self.has_fill and self.filled_shares >= self.requested_shares * 0.999


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

    def place_order(self, signal: "TradeSignal") -> Optional[OrderExecution]:
        """
        Sign and submit a limit order with automatic retry and urgency-aware order type.

        Order type selection:
          hours_to_resolution < IOC_URGENCY_HOURS → IOC (fill immediately or cancel)
          otherwise                                → GTC (Good-Till-Cancelled)

        Retries up to ORDER_RETRY_MAX times with exponential backoff (1s, 2s, 4s).
        Returns fill-aware execution details on success, None after all retries fail.
        """
        if not self.is_configured():
            return None

        for attempt in range(ORDER_RETRY_MAX):
            order_id = self._attempt_place_order(signal)
            if order_id is not None:
                return order_id
            if attempt < ORDER_RETRY_MAX - 1:
                delay = 2 ** attempt   # 1s → 2s → 4s
                logger.info(
                    "Order retry %d/%d for %s %s in %ds",
                    attempt + 2, ORDER_RETRY_MAX, signal.direction, signal.city, delay,
                )
                time.sleep(delay)

        logger.warning(
            "Order failed after %d attempts: %s %s EV=%.3f",
            ORDER_RETRY_MAX, signal.direction, signal.city, signal.ev,
        )
        return None

    def _attempt_place_order(self, signal: "TradeSignal") -> Optional[OrderExecution]:
        """Single order submission attempt. Returns execution details or None."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.constants import BUY

            if signal.direction == "YES":
                token_id = signal.market_id
                price = signal.market_price
            else:
                token_id = signal.no_token_id
                price = signal.market_price   # ev_calculator already flipped to NO price

            if not token_id:
                logger.warning(
                    "No token_id for %s %s — cannot place order",
                    signal.direction, signal.city,
                )
                return None

            price = round(max(0.001, min(0.999, price)), 4)
            requested_usd = round(signal.recommended_usd, 4)
            size = self._shares_for_usd(requested_usd, price)
            if size <= 0:
                logger.warning("Order size is zero for %s %s", signal.direction, signal.city)
                return None

            # Use IOC for urgent markets — fills immediately at best price or cancels.
            # Use GTC for markets with time — waits for a matching counterparty.
            order_type = (OrderType.IOC
                          if signal.hours_to_resolution < IOC_URGENCY_HOURS
                          else OrderType.GTC)

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,
            )
            signed_order = self._client.create_order(order_args)
            response = self._client.post_order(signed_order, order_type)

            order_id: Optional[str] = None
            if isinstance(response, dict):
                order_id = response.get("orderID") or response.get("order_id")
            elif hasattr(response, "order_id"):
                order_id = response.order_id

            if order_id:
                logger.info(
                    "LIVE ORDER [%s]: %s %s @ %.4f shares=%.4f notional≈$%.2f → %s",
                    order_type, signal.direction, signal.city, price, size, requested_usd, order_id,
                )
            else:
                logger.warning("Order posted but no order_id in response: %s", response)
                return None

            return self._execution_from_response(
                response=response,
                order_id=order_id,
                token_id=token_id,
                side=BUY,
                limit_price=price,
                requested_shares=size,
                requested_usd=requested_usd,
            )

        except Exception as exc:
            logger.warning(
                "place_order attempt failed for %s %s: %s",
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

    def place_exit_order(
        self,
        position: "Position",
        token_id: str,
        price: float,
    ) -> Optional[OrderExecution]:
        """
        Best-effort live exit for a locally tracked position.

        The local position manager measures `size_usd` as cost basis. CLOB sell
        size is token shares, so we convert cost basis to approximate shares
        using filled shares. Returns execution details on success, None on failure.
        """
        if not self.is_configured() or not token_id:
            return None
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.constants import SELL

            exit_price = round(max(0.001, min(0.999, price)), 4)
            shares = round(
                position.size_shares
                or (position.size_usd / max(position.entry_price, 0.001)),
                4,
            )
            order_args = OrderArgs(
                token_id=token_id,
                price=exit_price,
                size=shares,
                side=SELL,
            )
            signed_order = self._client.create_order(order_args)
            response = self._client.post_order(signed_order, OrderType.IOC)

            order_id: Optional[str] = None
            if isinstance(response, dict):
                order_id = response.get("orderID") or response.get("order_id")
            elif hasattr(response, "order_id"):
                order_id = response.order_id

            if order_id:
                logger.info(
                    "LIVE EXIT: %s %s @ %.4f shares=%.2f → %s",
                    position.direction, position.city, exit_price, shares, order_id,
                )
            else:
                logger.warning("Exit order posted but no order_id in response: %s", response)
                return None
            return self._execution_from_response(
                response=response,
                order_id=order_id,
                token_id=token_id,
                side=SELL,
                limit_price=exit_price,
                requested_shares=shares,
                requested_usd=round(shares * exit_price, 4),
            )

        except Exception as exc:
            logger.warning(
                "place_exit_order failed for %s %s: %s",
                position.direction, position.city, exc,
            )
            return None

    def _shares_for_usd(self, amount_usd: float, price: float) -> float:
        """Convert USDC notional into CLOB outcome-token shares."""
        if amount_usd <= 0 or price <= 0:
            return 0.0
        return round(amount_usd / price, 4)

    def _extract_order_id(self, response) -> Optional[str]:
        if isinstance(response, dict):
            return response.get("orderID") or response.get("order_id") or response.get("id")
        if hasattr(response, "order_id"):
            return response.order_id
        if hasattr(response, "orderID"):
            return response.orderID
        return None

    def _as_dict(self, value) -> dict:
        if isinstance(value, dict):
            return value
        if hasattr(value, "dict"):
            try:
                data = value.dict()
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {}

    def _numeric(self, data: dict, *keys: str) -> float:
        for key in keys:
            raw = data.get(key)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _order_snapshot(self, order_id: str) -> dict:
        if not self.is_configured() or not order_id:
            return {}
        try:
            raw = self._client.get_order(order_id)
            return self._as_dict(raw)
        except Exception as exc:
            logger.debug("get_order %s failed: %s", order_id, exc)
            return {}

    def _execution_from_response(
        self,
        response,
        order_id: str,
        token_id: str,
        side: str,
        limit_price: float,
        requested_shares: float,
        requested_usd: float,
    ) -> OrderExecution:
        data = self._as_dict(response)
        if order_id:
            snapshot = self._order_snapshot(order_id)
            if snapshot:
                data = {**data, **snapshot}

        status = str(data.get("status") or data.get("state") or "submitted").lower()
        filled_shares = self._numeric(
            data,
            "filled_size",
            "filledSize",
            "matched_size",
            "matchedSize",
            "size_matched",
            "sizeMatched",
            "filled",
        )
        remaining_shares = self._numeric(data, "remaining_size", "remainingSize", "size_remaining")
        original_shares = self._numeric(data, "original_size", "originalSize", "size")
        if filled_shares <= 0 and original_shares > 0 and remaining_shares > 0:
            filled_shares = max(0.0, original_shares - remaining_shares)

        average_price = self._numeric(
            data,
            "average_price",
            "averagePrice",
            "avg_price",
            "avgPrice",
            "price",
        ) or limit_price
        filled_usd = self._numeric(
            data,
            "filled_amount",
            "filledAmount",
            "matched_amount",
            "matchedAmount",
            "notional",
        )
        if filled_usd <= 0 and filled_shares > 0:
            filled_usd = filled_shares * average_price

        return OrderExecution(
            order_id=order_id,
            token_id=token_id,
            side=side,
            limit_price=limit_price,
            requested_shares=requested_shares,
            requested_usd=requested_usd,
            filled_shares=round(filled_shares, 6),
            filled_usd=round(filled_usd, 6),
            average_price=round(average_price, 6),
            status=status,
        )

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
