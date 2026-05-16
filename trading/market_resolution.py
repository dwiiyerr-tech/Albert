"""
Official Polymarket market resolution lookup.

The trading scanner stores CLOB token IDs as market identifiers. To settle a
position against Polymarket's official outcome, we first map the token to its
condition ID through CLOB, then fetch the Gamma market and inspect its final
outcome prices.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from config import POLYMARKET_BASE, POLYMARKET_GAMMA

logger = logging.getLogger(__name__)


@dataclass
class MarketResolution:
    market_id: str
    condition_id: str
    question: str
    outcome_yes: Optional[bool]
    yes_payout: float
    no_payout: float
    source: str = "polymarket_gamma"


class PolymarketResolutionClient:
    """Resolve CLOB token IDs to official Polymarket settlement payouts."""

    def __init__(
        self,
        gamma_base: str = POLYMARKET_GAMMA,
        clob_base: str = POLYMARKET_BASE,
        timeout_seconds: int = 10,
    ) -> None:
        self._gamma_base = gamma_base.rstrip("/")
        self._clob_base = clob_base.rstrip("/")
        self._timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._condition_cache: dict[str, dict] = {}
        self._market_cache: dict[str, Optional[dict]] = {}

    def _get_json(self, url: str, params: dict | None = None) -> Optional[dict | list]:
        try:
            resp = self._session.get(url, params=params or {}, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Polymarket resolution request failed: %s | %s", url, exc)
            return None

    def _coerce_list(self, value) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    def _token_mapping(self, token_id: str) -> Optional[dict]:
        if token_id in self._condition_cache:
            return self._condition_cache[token_id]
        data = self._get_json(f"{self._clob_base}/markets-by-token/{token_id}")
        if not isinstance(data, dict) or not data.get("condition_id"):
            return None
        self._condition_cache[token_id] = data
        return data

    def _gamma_market_by_condition(self, condition_id: str) -> Optional[dict]:
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]
        data = self._get_json(
            f"{self._gamma_base}/markets",
            params={"condition_ids": condition_id, "closed": "true", "limit": 1},
        )
        market = data[0] if isinstance(data, list) and data else None
        self._market_cache[condition_id] = market
        return market

    def _payouts_from_market(self, market: dict) -> Optional[tuple[float, float]]:
        if market.get("closed") is not True:
            return None

        status = str(market.get("umaResolutionStatus") or "").lower()
        prices = self._coerce_list(market.get("outcomePrices"))
        outcomes = [str(o).lower() for o in self._coerce_list(market.get("outcomes"))]
        if len(prices) < 2:
            return None

        try:
            yes_idx = outcomes.index("yes") if "yes" in outcomes else 0
            no_idx = outcomes.index("no") if "no" in outcomes else 1
            yes = float(prices[yes_idx])
            no = float(prices[no_idx])
        except (TypeError, ValueError, IndexError):
            return None

        # Current resolved Gamma weather markets expose exact "0"/"1" prices
        # with umaResolutionStatus="resolved". Older markets may omit status but
        # still expose decisive final outcome prices.
        resolved_status = status in {"resolved", "settled", "finalized"}
        binary_decisive = (
            (yes >= 0.99 and no <= 0.01)
            or (no >= 0.99 and yes <= 0.01)
        )
        half_resolution = (
            resolved_status
            and 0.49 <= yes <= 0.51
            and 0.49 <= no <= 0.51
        )
        decisive = binary_decisive or half_resolution
        if status and not resolved_status and not decisive:
            return None
        if not decisive:
            return None

        return max(0.0, min(1.0, yes)), max(0.0, min(1.0, no))

    def resolve_token(self, token_id: str) -> Optional[MarketResolution]:
        """
        Return official settlement payouts for a CLOB token ID, or None when the
        market is not resolved yet or cannot be resolved safely from public data.
        """
        if not token_id or token_id.startswith("demo:"):
            return None

        mapping = self._token_mapping(token_id)
        if not mapping:
            return None

        condition_id = str(mapping["condition_id"])
        market = self._gamma_market_by_condition(condition_id)
        if not market:
            return None

        payouts = self._payouts_from_market(market)
        if not payouts:
            return None

        yes_payout, no_payout = payouts
        if yes_payout > no_payout:
            outcome_yes: Optional[bool] = True
        elif no_payout > yes_payout:
            outcome_yes = False
        else:
            outcome_yes = None

        return MarketResolution(
            market_id=str(market.get("id") or token_id),
            condition_id=condition_id,
            question=str(market.get("question") or ""),
            outcome_yes=outcome_yes,
            yes_payout=yes_payout,
            no_payout=no_payout,
        )

    def token_pair(self, token_id: str) -> Optional[tuple[str, str]]:
        """Return (YES token id, NO token id) for a CLOB token id."""
        mapping = self._token_mapping(token_id)
        if not mapping:
            return None
        yes = str(mapping.get("primary_token_id") or "")
        no = str(mapping.get("secondary_token_id") or "")
        if not yes or not no:
            return None
        return yes, no
