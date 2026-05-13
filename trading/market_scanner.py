"""
Polymarket Scanner — from WeatherBot (alteregoeth-ai/weatherbot)

Execution upgrades:
  - requests.Session with connection pooling (persistent TCP, keep-alive)
  - Parallel order-book fetching via ThreadPoolExecutor
  - Token-bucket rate limiter to stay within Polymarket API limits
"""
from __future__ import annotations

import datetime
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from config import POLYMARKET_BASE, POLYMARKET_GAMMA, MAX_PARALLEL_ORDERBOOKS
from weather_data import parse_temp_range

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple token-bucket rate limiter — thread-safe."""

    def __init__(self, calls_per_second: float = 5.0) -> None:
        self._interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            gap = self._interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


class MarketScanner:
    """Fetches and filters Polymarket weather markets."""

    _WEATHER_KEYWORDS = ["temperature", "temp", "degrees", "°f", "high", "low",
                          "weather", "forecast"]

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        # Adapters for connection pooling (pool_connections=10 TCP connections)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._rate = _RateLimiter(calls_per_second=5.0)

    def _get(self, url: str, params: dict = None, timeout: int = 10) -> Optional[dict | list]:
        self._rate.wait()
        try:
            resp = self._session.get(url, params=params or {}, timeout=timeout)
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

    def _fetch_market_entry(self, market: dict) -> Optional[dict]:
        """
        Fetch the live order book for one market and return a complete market dict,
        or None if the market should be skipped.
        Runs in parallel threads — no shared-state writes.
        """
        question = market.get("question", "")
        bucket = parse_temp_range(question)
        if not bucket:
            return None

        clob_token_ids = market.get("clobTokenIds", [""])
        clob_token_id = clob_token_ids[0] if clob_token_ids else ""
        no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else ""
        if not clob_token_id:
            return None

        ob = self._get(f"{POLYMARKET_BASE}/book", params={"token_id": clob_token_id})
        if not ob:
            return None

        try:
            asks = ob.get("asks") or []
            bids = ob.get("bids") or []
            best_ask = float(asks[0].get("price", 1.0)) if asks else 1.0
            best_bid = float(bids[0].get("price", 0.0)) if bids else 0.0
            spread = best_ask - best_bid
            if spread < 0:
                return None
            mid = max(0.01, min(0.99, (best_ask + best_bid) / 2))
            volume = float(market.get("volume", 0))
            hours = self._hours_until_resolution(market.get("endDate", ""))
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("Order book parse failed: %s", exc)
            return None

        b_low, b_high = bucket[0], bucket[1]
        if b_low != float("-inf") and b_high != float("inf") and b_low >= b_high:
            b_low, b_high = b_high, b_low

        return {
            "market_id": clob_token_id,    # CLOB YES token ID for OrderArgs
            "no_token_id": no_token_id,     # CLOB NO token ID
            "question": question,
            "price_yes": mid,
            "price_no": 1 - mid,
            "spread": spread,
            "volume": volume,
            "hours_to_resolution": hours,
            "bucket_low": b_low,
            "bucket_high": b_high,
        }

    def get_open_markets(self, city_name: str) -> list[dict]:
        """
        Return open Polymarket markets mentioning a city and temperature keywords.
        Order books are fetched in parallel for lower latency.

        Each result dict: {market_id, no_token_id, question, price_yes, price_no,
                           spread, volume, hours_to_resolution, bucket_low, bucket_high}
        """
        data = self._get(
            f"{POLYMARKET_GAMMA}/events",
            params={"q": city_name, "active": "true", "limit": 50},
        )
        if not data:
            return []

        events = data if isinstance(data, list) else data.get("data", [])

        # Collect candidate markets from all events (no HTTP yet)
        candidates: list[dict] = []
        for event in events:
            if not self._is_weather_market(event.get("title", "")):
                continue
            for market in event.get("markets", []):
                if parse_temp_range(market.get("question", "")):
                    candidates.append(market)

        if not candidates:
            return []

        # Fetch all order books in parallel
        markets: list[dict] = []
        max_workers = min(len(candidates), MAX_PARALLEL_ORDERBOOKS)
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="orderbook") as pool:
            futures = {pool.submit(self._fetch_market_entry, m): m for m in candidates}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        markets.append(result)
                except Exception as exc:
                    logger.debug("Market fetch error: %s", exc)

        logger.info("Found %d weather markets for %s", len(markets), city_name)
        return markets
