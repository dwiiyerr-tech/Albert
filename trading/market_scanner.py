"""
Polymarket Scanner — from WeatherBot (alteregoeth-ai/weatherbot)

Execution upgrades:
  - requests.Session with connection pooling (persistent TCP, keep-alive)
  - Parallel order-book fetching via ThreadPoolExecutor
  - Token-bucket rate limiter to stay within Polymarket API limits
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from config import (
    MARKET_SCANNER_DEBUG,
    POLYMARKET_BASE,
    POLYMARKET_GAMMA,
    MAX_PARALLEL_ORDERBOOKS,
    MAX_ORDERBOOK_SLIPPAGE,
    MIN_ORDERBOOK_DEPTH_USD,
)
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

    def _effective_ask(self, asks: list[dict]) -> Optional[dict]:
        levels: list[tuple[float, float]] = []
        for ask in asks:
            try:
                price = float(ask.get("price", 0.0))
                size = float(ask.get("size", 0.0))
            except (TypeError, ValueError):
                continue
            if 0 < price < 1 and size > 0:
                levels.append((price, size))

        if not levels:
            return None

        levels.sort(key=lambda item: item[0])
        best_ask = levels[0][0]
        max_fill_price = min(0.99, best_ask + MAX_ORDERBOOK_SLIPPAGE)
        fill_levels = [(price, size) for price, size in levels if price <= max_fill_price]
        depth_usd = sum(price * size for price, size in fill_levels)
        if depth_usd < MIN_ORDERBOOK_DEPTH_USD:
            return None

        remaining = MIN_ORDERBOOK_DEPTH_USD
        total_cost = 0.0
        total_shares = 0.0
        for price, size in fill_levels:
            level_cost = price * size
            take_cost = min(remaining, level_cost)
            if take_cost <= 0:
                continue
            total_cost += take_cost
            total_shares += take_cost / price
            remaining -= take_cost
            if remaining <= 1e-9:
                break

        if total_shares <= 0:
            return None
        effective_ask = total_cost / total_shares
        return {
            "best_ask": max(0.01, min(0.99, best_ask)),
            "effective_ask": max(0.01, min(0.99, effective_ask)),
            "ask_depth_usd": depth_usd,
            "slippage": max(0.0, effective_ask - best_ask),
        }

    def _book_prices(self, token_id: str) -> Optional[dict]:
        ob = self._get(f"{POLYMARKET_BASE}/book", params={"token_id": token_id})
        if not ob:
            return None
        try:
            asks = ob.get("asks") or []
            bids = ob.get("bids") or []
            ask = self._effective_ask(asks)
            if not ask:
                return None
            best_bid = max(float(b.get("price", 0.0)) for b in bids) if bids else 0.0
            spread = ask["best_ask"] - best_bid
            if spread < 0:
                return None
            return {
                "best_ask": ask["best_ask"],
                "effective_ask": ask["effective_ask"],
                "best_bid": max(0.01, min(0.99, best_bid)),
                "mid": max(0.01, min(0.99, (ask["best_ask"] + best_bid) / 2)),
                "spread": spread,
                "ask_depth_usd": ask["ask_depth_usd"],
                "slippage": ask["slippage"],
            }
        except (ValueError, TypeError) as exc:
            logger.debug("Order book parse failed: %s", exc)
            return None

    def get_token_exit_price(self, token_id: str) -> Optional[float]:
        """
        Return the current liquidation price for a long outcome token.

        We use best bid because an exit from a long YES/NO token sells into bids.
        The scanner's entry logic uses effective ask; this method is deliberately
        conservative for mark-to-market and stop-loss checks.
        """
        if not token_id:
            return None
        book = self._book_prices(token_id)
        if not book:
            return None
        return book["best_bid"]

    def _is_weather_market(self, title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in self._WEATHER_KEYWORDS)

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

    def _market_end_date(self, market: dict) -> Optional[datetime.date]:
        end_date = market.get("endDate") or market.get("endDateIso") or market.get("end_date_iso")
        if not end_date:
            return None
        try:
            return datetime.datetime.fromisoformat(
                str(end_date).replace("Z", "+00:00")
            ).date()
        except Exception:
            return None

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

        if market.get("closed") is True or market.get("active") is False:
            return None
        if market.get("enableOrderBook") is False:
            return None

        clob_token_ids = self._coerce_list(market.get("clobTokenIds", [""]))
        clob_token_id = clob_token_ids[0] if clob_token_ids else ""
        no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else ""
        if not clob_token_id:
            return None

        yes_book = self._book_prices(clob_token_id)
        if not yes_book:
            return None
        no_book = self._book_prices(no_token_id) if no_token_id else None

        try:
            price_yes = yes_book["effective_ask"]
            price_no = no_book["effective_ask"] if no_book else max(0.01, min(0.99, 1 - yes_book["best_bid"]))
            spread = max(yes_book["spread"], no_book["spread"] if no_book else yes_book["spread"])
            slippage = max(yes_book["slippage"], no_book["slippage"] if no_book else yes_book["slippage"])
            depth_usd = min(
                yes_book["ask_depth_usd"],
                no_book["ask_depth_usd"] if no_book else yes_book["ask_depth_usd"],
            )
            volume = float(market.get("volume", 0))
            hours = self._hours_until_resolution(market.get("endDate", ""))
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            logger.debug("Order book parse failed: %s", exc)
            return None

        b_low, b_high = bucket[0], bucket[1]
        if b_low != float("-inf") and b_high != float("inf") and b_low >= b_high:
            b_low, b_high = b_high, b_low

        return {
            "market_id": clob_token_id,    # CLOB YES token ID for OrderArgs
            "no_token_id": no_token_id,     # CLOB NO token ID
            "condition_id": str(market.get("conditionId") or market.get("condition_id") or ""),
            "question": question,
            "price_yes": price_yes,
            "price_no": price_no,
            "mid_yes": yes_book["mid"],
            "best_bid_yes": yes_book["best_bid"],
            "spread": spread,
            "slippage": slippage,
            "orderbook_depth_usd": depth_usd,
            "volume": volume,
            "hours_to_resolution": hours,
            "bucket_low": b_low,
            "bucket_high": b_high,
        }

    def _candidate_markets_from_public_search(
        self,
        city_name: str,
        target_date: datetime.date | None,
    ) -> list[dict]:
        data = self._get(
            f"{POLYMARKET_GAMMA}/public-search",
            params={
                "q": f"temperature {city_name}",
                "active": "true",
                "closed": "false",
                "limit": 50,
            },
        )
        if not data:
            return []

        candidates: list[dict] = []
        skipped = {"event_closed": 0, "wrong_date": 0, "not_weather": 0, "no_temp_range": 0}
        for event in data.get("events", []):
            if event.get("closed") is True or event.get("active") is False:
                skipped["event_closed"] += 1
                continue
            title = event.get("title", "")
            if not self._is_weather_market(title):
                skipped["not_weather"] += 1
                continue
            for market in event.get("markets", []):
                if target_date and self._market_end_date(market) != target_date:
                    skipped["wrong_date"] += 1
                    continue
                if parse_temp_range(market.get("question", "")):
                    candidates.append(market)
                else:
                    skipped["no_temp_range"] += 1

        if MARKET_SCANNER_DEBUG:
            logger.info(
                "Market public-search %s: %d candidates (skipped=%s)",
                city_name,
                len(candidates),
                skipped,
            )
        return candidates

    def get_open_markets(
        self,
        city_name: str,
        target_date: datetime.date | None = None,
    ) -> list[dict]:
        """
        Return open Polymarket markets mentioning a city and temperature keywords.
        Order books are fetched in parallel for lower latency.

        Each result dict: {market_id, no_token_id, question, price_yes, price_no,
                           spread, volume, hours_to_resolution, bucket_low, bucket_high}
        """
        candidates = self._candidate_markets_from_public_search(city_name, target_date)

        data = self._get(
            f"{POLYMARKET_GAMMA}/events",
            params={"q": city_name, "active": "true", "closed": "false", "limit": 50},
        )
        if not data and not candidates:
            if MARKET_SCANNER_DEBUG:
                logger.info("Market scan %s: no Gamma events returned", city_name)
            return []

        events = data if isinstance(data, list) else data.get("data", []) if data else []
        if MARKET_SCANNER_DEBUG:
            logger.info("Market scan %s: %d Gamma events", city_name, len(events))

        # Collect candidate markets from all events (no HTTP yet)
        skipped = {"event_closed": 0, "wrong_date": 0, "not_weather": 0, "no_temp_range": 0}
        for event in events:
            if event.get("closed") is True or event.get("active") is False:
                skipped["event_closed"] += 1
                continue
            if not self._is_weather_market(event.get("title", "")):
                skipped["not_weather"] += 1
                continue
            for market in event.get("markets", []):
                if target_date and self._market_end_date(market) != target_date:
                    skipped["wrong_date"] += 1
                    continue
                question = market.get("question", "")
                if parse_temp_range(question):
                    candidates.append(market)
                else:
                    skipped["no_temp_range"] += 1

        if not candidates:
            if MARKET_SCANNER_DEBUG:
                logger.info(
                    "Market scan %s: no candidates (skipped=%s)",
                    city_name, skipped,
                )
            return []
        if MARKET_SCANNER_DEBUG:
            logger.info("Market scan %s: %d candidates (skipped=%s)",
                        city_name, len(candidates), skipped)

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
