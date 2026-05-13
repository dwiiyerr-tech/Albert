"""
Polymarket Scanner — from WeatherBot (alteregoeth-ai/weatherbot)

Queries the Polymarket CLOB and Gamma APIs for temperature prediction markets,
then feeds them to the EV calculator with simulation probabilities.
"""
from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Optional

import requests

from config import POLYMARKET_BASE, POLYMARKET_GAMMA
from weather_data import parse_temp_range

logger = logging.getLogger(__name__)


class MarketScanner:
    """Fetches and filters Polymarket weather markets."""

    _WEATHER_KEYWORDS = ["temperature", "temp", "degrees", "°f", "high", "low",
                          "weather", "forecast"]

    def _get(self, url: str, params: dict = None, timeout: int = 10) -> Optional[dict | list]:
        try:
            resp = requests.get(url, params=params or {}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Polymarket request failed: %s | %s", url, exc)
            return None

    def _is_weather_market(self, title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in self._WEATHER_KEYWORDS)

    def _hours_until_resolution(self, end_date_iso: str) -> float:
        try:
            end = datetime.datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = (end - now).total_seconds() / 3600
            return max(0.0, delta)
        except Exception:
            return 0.0

    def get_open_markets(self, city_name: str) -> list[dict]:
        """
        Return open Polymarket markets mentioning a city and temperature keywords.
        Each result dict: {market_id, question, price_yes, price_no, spread, volume,
                           hours_to_resolution, bucket_low, bucket_high}
        """
        city_slug = city_name.lower().replace(" ", "-")
        markets = []

        # Search Gamma API for events containing city name
        data = self._get(
            f"{POLYMARKET_GAMMA}/events",
            params={"q": city_name, "active": "true", "limit": 50},
        )
        if not data:
            return markets

        events = data if isinstance(data, list) else data.get("data", [])
        for event in events:
            title = event.get("title", "")
            if not self._is_weather_market(title):
                continue
            for market in event.get("markets", []):
                question = market.get("question", "")
                bucket = parse_temp_range(question)
                if not bucket:
                    continue

                # Fetch live order book for price
                clob_token_ids = market.get("clobTokenIds", [""])
                clob_token_id = clob_token_ids[0] if clob_token_ids else ""
                no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else ""
                ob = self._get(f"{POLYMARKET_BASE}/book", params={"token_id": clob_token_id})
                if not ob:
                    continue

                try:
                    asks = ob.get("asks") or []
                    bids = ob.get("bids") or []
                    best_ask = float(asks[0].get("price", 1.0)) if asks else 1.0
                    best_bid = float(bids[0].get("price", 0.0)) if bids else 0.0
                    spread = best_ask - best_bid
                    if spread < 0:
                        logger.debug("Skipping market with negative spread: %s",
                                     market.get("id", ""))
                        continue
                    mid = max(0.01, min(0.99, (best_ask + best_bid) / 2))
                    volume = float(market.get("volume", 0))
                    end_date = market.get("endDate", "")
                    hours = self._hours_until_resolution(end_date)
                except (ValueError, TypeError, IndexError) as exc:
                    logger.debug("Order book parse failed for market %s: %s",
                                 market.get("id", ""), exc)
                    continue

                # Validate bucket: low must be less than high
                b_low, b_high = bucket[0], bucket[1]
                if b_low != float("-inf") and b_high != float("inf") and b_low >= b_high:
                    logger.debug("Invalid bucket %s–%s for market %s, swapping",
                                 b_low, b_high, market.get("id", ""))
                    b_low, b_high = b_high, b_low

                markets.append({
                    "market_id": clob_token_id,   # CLOB YES token ID for OrderArgs
                    "no_token_id": no_token_id,    # CLOB NO token ID
                    "question": question,
                    "price_yes": mid,
                    "price_no": 1 - mid,
                    "spread": spread,
                    "volume": volume,
                    "hours_to_resolution": hours,
                    "bucket_low": b_low,
                    "bucket_high": b_high,
                })

        logger.info("Found %d weather markets for %s", len(markets), city_name)
        return markets
