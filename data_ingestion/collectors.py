from __future__ import annotations

import datetime as dt
import json
import logging
import time
from typing import Callable, Iterable, Optional

import requests

from config import (
    AVIATION_WEATHER_BASE,
    OPEN_METEO_BASE,
    POLYMARKET_BASE,
    POLYMARKET_GAMMA,
)
from .storage import ms_to_iso, now_utc_iso, utc_ms

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"


def _coerce_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _coerce_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_any(data: dict, *keys: str):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _unix_to_iso(value) -> str:
    if value in (None, ""):
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds > 10_000_000_000:
        seconds = seconds / 1000
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).isoformat()


def _join_csv(values: Iterable[str] | None) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return ",".join(str(value) for value in values if value)


def flatten_polymarket_events(payload: dict | list, collected_at: str | None = None) -> list[dict]:
    """Flatten Gamma public-search/events responses into market rows."""
    ts = collected_at or now_utc_iso()
    if isinstance(payload, dict):
        events = payload.get("events") or payload.get("data") or []
    elif isinstance(payload, list):
        events = payload
    else:
        events = []

    rows: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = str(_get_any(event, "id", "eventId") or "")
        event_title = str(_get_any(event, "title", "question", "name") or "")
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            tokens = [str(x) for x in _coerce_list(market.get("clobTokenIds"))]
            rows.append({
                "collected_at": ts,
                "event_id": event_id,
                "event_title": event_title,
                "event_slug": str(event.get("slug") or ""),
                "gamma_market_id": str(_get_any(market, "id", "marketId") or ""),
                "condition_id": str(_get_any(market, "conditionId", "condition_id") or ""),
                "question": str(market.get("question") or ""),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "end_date": str(_get_any(market, "endDate", "endDateIso", "end_date_iso") or ""),
                "volume": _coerce_float(_get_any(market, "volume", "volumeNum")),
                "liquidity": _coerce_float(_get_any(market, "liquidity", "liquidityNum")),
                "outcomes": _coerce_list(market.get("outcomes")),
                "outcome_prices": _coerce_list(market.get("outcomePrices")),
                "clob_token_ids": tokens,
                "yes_token_id": tokens[0] if tokens else "",
                "no_token_id": tokens[1] if len(tokens) > 1 else "",
                "raw_tags": _coerce_list(market.get("tags")),
            })
    return rows


def select_polymarket_targets(market_rows: Iterable[dict], max_markets: int = 20) -> dict:
    """Return high-liquidity condition/token targets from flattened market rows."""
    candidates: list[tuple[float, dict]] = []
    for row in market_rows:
        condition_id = str(row.get("condition_id") or "")
        yes_token_id = str(row.get("yes_token_id") or "")
        no_token_id = str(row.get("no_token_id") or "")
        if not condition_id and not yes_token_id and not no_token_id:
            continue
        volume = _coerce_float(row.get("volume")) or 0.0
        liquidity = _coerce_float(row.get("liquidity")) or 0.0
        candidates.append((volume + liquidity, row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _, row in candidates[:max(0, max_markets)]]

    condition_ids: list[str] = []
    token_ids: list[str] = []
    for row in selected:
        condition_id = str(row.get("condition_id") or "")
        if condition_id and condition_id not in condition_ids:
            condition_ids.append(condition_id)
        for key in ("yes_token_id", "no_token_id"):
            token_id = str(row.get(key) or "")
            if token_id and token_id not in token_ids:
                token_ids.append(token_id)

    return {
        "markets": selected,
        "condition_ids": condition_ids,
        "token_ids": token_ids,
    }


def rows_from_openmeteo_daily(
    data: dict,
    city: dict,
    source_model: str,
    collected_at: str | None = None,
) -> list[dict]:
    ts = collected_at or now_utc_iso()
    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    rows: list[dict] = []
    for idx, day in enumerate(dates):
        rows.append({
            "collected_at": ts,
            "city": city["name"],
            "lat": city["lat"],
            "lon": city["lon"],
            "model": source_model,
            "date": day,
            "temperature_2m_max_c": _nth(daily.get("temperature_2m_max"), idx),
            "temperature_2m_min_c": _nth(daily.get("temperature_2m_min"), idx),
            "temperature_2m_mean_c": _nth(daily.get("temperature_2m_mean"), idx),
            "precipitation_sum_mm": _nth(daily.get("precipitation_sum"), idx),
            "wind_speed_10m_max_kmh": _nth(daily.get("wind_speed_10m_max"), idx),
        })
    return rows


def rows_from_binance_klines(
    rows: Iterable[list],
    symbol: str,
    interval: str,
    collected_at: str | None = None,
) -> list[dict]:
    ts = collected_at or now_utc_iso()
    parsed: list[dict] = []
    for row in rows:
        if len(row) < 11:
            continue
        parsed.append({
            "collected_at": ts,
            "symbol": symbol,
            "interval": interval,
            "open_time": ms_to_iso(row[0]),
            "open_time_ms": int(row[0]),
            "open": _coerce_float(row[1]),
            "high": _coerce_float(row[2]),
            "low": _coerce_float(row[3]),
            "close": _coerce_float(row[4]),
            "volume": _coerce_float(row[5]),
            "close_time": ms_to_iso(row[6]),
            "close_time_ms": int(row[6]),
            "quote_volume": _coerce_float(row[7]),
            "trade_count": int(row[8]),
            "taker_buy_base_volume": _coerce_float(row[9]),
            "taker_buy_quote_volume": _coerce_float(row[10]),
        })
    return parsed


def rows_from_polymarket_price_history(
    payload: dict | list,
    token_id: str,
    interval: str,
    collected_at: str | None = None,
) -> list[dict]:
    ts = collected_at or now_utc_iso()
    if isinstance(payload, dict):
        points = payload.get("history") or payload.get("prices") or payload.get("data") or []
    elif isinstance(payload, list):
        points = payload
    else:
        points = []

    rows: list[dict] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        timestamp = _get_any(point, "t", "timestamp", "time")
        rows.append({
            "collected_at": ts,
            "token_id": token_id,
            "interval": interval,
            "timestamp": timestamp,
            "timestamp_iso": _unix_to_iso(timestamp),
            "price": _coerce_float(_get_any(point, "p", "price")),
            "raw": point,
        })
    return rows


def rows_from_polymarket_trades(payload: dict | list, collected_at: str | None = None) -> list[dict]:
    ts = collected_at or now_utc_iso()
    trades = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(trades, list):
        return []

    rows: list[dict] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        timestamp = _get_any(trade, "timestamp", "createdAt", "created_at")
        rows.append({
            "collected_at": ts,
            "proxy_wallet": str(_get_any(trade, "proxyWallet", "proxy_wallet") or ""),
            "side": str(trade.get("side") or ""),
            "asset": str(trade.get("asset") or ""),
            "condition_id": str(_get_any(trade, "conditionId", "condition_id") or ""),
            "size": _coerce_float(trade.get("size")),
            "price": _coerce_float(trade.get("price")),
            "timestamp": timestamp,
            "timestamp_iso": _unix_to_iso(timestamp),
            "title": str(trade.get("title") or ""),
            "slug": str(trade.get("slug") or ""),
            "event_slug": str(_get_any(trade, "eventSlug", "event_slug") or ""),
            "outcome": str(trade.get("outcome") or ""),
            "outcome_index": _get_any(trade, "outcomeIndex", "outcome_index"),
            "name": str(trade.get("name") or ""),
            "pseudonym": str(trade.get("pseudonym") or ""),
            "transaction_hash": str(_get_any(trade, "transactionHash", "transaction_hash") or ""),
            "raw": trade,
        })
    return rows


def rows_from_polymarket_holders(payload: dict | list, collected_at: str | None = None) -> list[dict]:
    ts = collected_at or now_utc_iso()
    markets = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(markets, list):
        return []

    rows: list[dict] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        token_id = str(_get_any(market, "token", "asset", "token_id") or "")
        for holder in market.get("holders") or []:
            if not isinstance(holder, dict):
                continue
            rows.append({
                "collected_at": ts,
                "token_id": token_id,
                "proxy_wallet": str(_get_any(holder, "proxyWallet", "proxy_wallet") or ""),
                "asset": str(holder.get("asset") or ""),
                "amount": _coerce_float(holder.get("amount")),
                "outcome_index": _get_any(holder, "outcomeIndex", "outcome_index"),
                "name": str(holder.get("name") or ""),
                "pseudonym": str(holder.get("pseudonym") or ""),
                "raw": holder,
            })
    return rows


def rows_from_polymarket_open_interest(payload: dict | list, collected_at: str | None = None) -> list[dict]:
    ts = collected_at or now_utc_iso()
    markets = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(markets, list):
        return []

    rows: list[dict] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        rows.append({
            "collected_at": ts,
            "condition_id": str(_get_any(market, "market", "conditionId", "condition_id") or ""),
            "open_interest": _coerce_float(_get_any(market, "value", "openInterest", "open_interest")),
            "raw": market,
        })
    return rows


def rows_from_polymarket_activity(payload: dict | list, collected_at: str | None = None) -> list[dict]:
    ts = collected_at or now_utc_iso()
    activity = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(activity, list):
        return []

    rows: list[dict] = []
    for item in activity:
        if not isinstance(item, dict):
            continue
        timestamp = _get_any(item, "timestamp", "createdAt", "created_at")
        rows.append({
            "collected_at": ts,
            "proxy_wallet": str(_get_any(item, "proxyWallet", "proxy_wallet") or ""),
            "condition_id": str(_get_any(item, "conditionId", "condition_id") or ""),
            "asset": str(item.get("asset") or ""),
            "type": str(item.get("type") or ""),
            "side": str(item.get("side") or ""),
            "size": _coerce_float(item.get("size")),
            "usdc_size": _coerce_float(_get_any(item, "usdcSize", "usdc_size")),
            "price": _coerce_float(item.get("price")),
            "timestamp": timestamp,
            "timestamp_iso": _unix_to_iso(timestamp),
            "transaction_hash": str(_get_any(item, "transactionHash", "transaction_hash") or ""),
            "raw": item,
        })
    return rows


def _nth(values, idx: int):
    if not isinstance(values, list) or idx >= len(values):
        return None
    return values[idx]


class DataCollector:
    """Small no-key data collector for Albert research and replay datasets."""

    def __init__(
        self,
        session: requests.Session | None = None,
        sleep_seconds: float = 0.3,
        get_json: Callable[[str, dict], object] | None = None,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._sleep_seconds = sleep_seconds
        self._get_json_override = get_json

    def get_json(self, url: str, params: dict | None = None, timeout: int = 20):
        if self._get_json_override:
            return self._get_json_override(url, params or {})
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                resp = self._session.get(url, params=params or {}, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else 0
                if attempt < 5 and status == 429:
                    time.sleep(5 * attempt)
                    continue
                if attempt < 5 and 500 <= status < 600:
                    time.sleep(attempt)
                    continue
            except Exception as exc:
                last_exc = exc
                if attempt < 5:
                    time.sleep(attempt)
        logger.warning("Data request failed: %s params=%s error=%s", url, params, last_exc)
        return None

    def collect_polymarket_markets(self, query: str = "temperature", limit: int = 100) -> list[dict]:
        payload = self.get_json(
            f"{POLYMARKET_GAMMA}/public-search",
            {"q": query, "active": "true", "closed": "false", "limit": limit},
        )
        return flatten_polymarket_events(payload or {})

    def collect_polymarket_orderbooks(self, token_ids: Iterable[str]) -> list[dict]:
        rows: list[dict] = []
        for token_id in token_ids:
            if not token_id:
                continue
            payload = self.get_json(f"{POLYMARKET_BASE}/book", {"token_id": token_id})
            if isinstance(payload, dict):
                rows.append({
                    "collected_at": now_utc_iso(),
                    "token_id": token_id,
                    "bids": payload.get("bids") or [],
                    "asks": payload.get("asks") or [],
                    "min_order_size": payload.get("min_order_size"),
                    "tick_size": payload.get("tick_size"),
                })
            time.sleep(self._sleep_seconds)
        return rows

    def collect_polymarket_price_history(
        self,
        token_ids: Iterable[str],
        start_date: dt.date,
        end_date: dt.date,
        interval: str = "1d",
        fidelity: int | None = None,
    ) -> list[dict]:
        rows: list[dict] = []
        start_ts = utc_ms(start_date) // 1000
        end_ts = utc_ms(end_date, end_of_day=True) // 1000
        for token_id in token_ids:
            if not token_id:
                continue
            params = {
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": interval,
            }
            if fidelity is not None:
                params["fidelity"] = fidelity
            payload = self.get_json(f"{POLYMARKET_BASE}/prices-history", params)
            rows.extend(rows_from_polymarket_price_history(payload or {}, token_id, interval))
            time.sleep(self._sleep_seconds)
        return rows

    def collect_polymarket_trades(
        self,
        condition_ids: Iterable[str] | None = None,
        limit: int = 100,
        offset: int = 0,
        taker_only: bool = True,
    ) -> list[dict]:
        params: dict = {"limit": limit, "offset": offset, "takerOnly": str(taker_only).lower()}
        market = _join_csv(condition_ids)
        if market:
            params["market"] = market
        payload = self.get_json(f"{POLYMARKET_DATA_API}/trades", params)
        return rows_from_polymarket_trades(payload or {})

    def collect_polymarket_holders(
        self,
        condition_ids: Iterable[str],
        limit: int = 20,
        min_balance: int = 1,
    ) -> list[dict]:
        market = _join_csv(condition_ids)
        if not market:
            return []
        payload = self.get_json(
            f"{POLYMARKET_DATA_API}/holders",
            {"market": market, "limit": min(limit, 20), "minBalance": min_balance},
        )
        return rows_from_polymarket_holders(payload or {})

    def collect_polymarket_open_interest(self, condition_ids: Iterable[str] | None = None) -> list[dict]:
        params = {}
        market = _join_csv(condition_ids)
        if market:
            params["market"] = market
        payload = self.get_json(f"{POLYMARKET_DATA_API}/oi", params)
        return rows_from_polymarket_open_interest(payload or {})

    def collect_polymarket_user_activity(
        self,
        user: str,
        condition_ids: Iterable[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        params: dict = {"user": user, "limit": limit, "offset": offset}
        market = _join_csv(condition_ids)
        if market:
            params["market"] = market
        payload = self.get_json(f"{POLYMARKET_DATA_API}/activity", params)
        return rows_from_polymarket_activity(payload or {})

    def collect_weather_forecast(self, cities: Iterable[dict], days: int = 7) -> list[dict]:
        rows: list[dict] = []
        daily = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
        for city in cities:
            for model in ("ecmwf_ifs025", "gfs_seamless"):
                payload = self.get_json(
                    f"{OPEN_METEO_BASE}/forecast",
                    {
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "daily": daily,
                        "temperature_unit": "celsius",
                        "forecast_days": days,
                        "models": model,
                        "timezone": "auto",
                    },
                )
                if isinstance(payload, dict):
                    rows.extend(rows_from_openmeteo_daily(payload, city, model))
                time.sleep(self._sleep_seconds)
        return rows

    def collect_weather_actuals(
        self,
        cities: Iterable[dict],
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[dict]:
        rows: list[dict] = []
        daily = (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,wind_speed_10m_max"
        )
        for city in cities:
            payload = self.get_json(
                OPEN_METEO_ARCHIVE,
                {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "daily": daily,
                    "temperature_unit": "celsius",
                    "timezone": "auto",
                },
            )
            if isinstance(payload, dict):
                rows.extend(rows_from_openmeteo_daily(payload, city, "openmeteo_archive"))
            time.sleep(self._sleep_seconds)
        return rows

    def collect_metar(self, cities: Iterable[dict]) -> list[dict]:
        rows: list[dict] = []
        for city in cities:
            icao = city.get("metar")
            if not icao:
                continue
            payload = self.get_json(f"{AVIATION_WEATHER_BASE}/metar", {"ids": icao, "format": "json"})
            observations = payload if isinstance(payload, list) else []
            for obs in observations:
                if isinstance(obs, dict):
                    rows.append({
                        "collected_at": now_utc_iso(),
                        "city": city["name"],
                        "icao": icao,
                        "raw_observation": obs.get("rawOb") or obs.get("raw_observation") or "",
                        "obs_time": obs.get("obsTime") or obs.get("reportTime") or "",
                        "temp_c": _coerce_float(obs.get("temp") or obs.get("tempC")),
                        "raw": obs,
                    })
            time.sleep(self._sleep_seconds)
        return rows

    def collect_binance_klines(
        self,
        symbol: str,
        interval: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[dict]:
        rows: list[dict] = []
        start_ms = utc_ms(start_date)
        end_ms = utc_ms(end_date, end_of_day=True)
        while start_ms <= end_ms:
            payload = self.get_json(
                f"{BINANCE_SPOT_BASE}/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list) or not payload:
                break
            parsed = rows_from_binance_klines(payload, symbol, interval)
            rows.extend(parsed)
            last_close_ms = int(payload[-1][6])
            next_start = last_close_ms + 1
            if next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(self._sleep_seconds)
        return rows

    def collect_binance_funding(
        self,
        symbol: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[dict]:
        rows: list[dict] = []
        start_ms = utc_ms(start_date)
        end_ms = utc_ms(end_date, end_of_day=True)
        while start_ms <= end_ms:
            payload = self.get_json(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            )
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                rows.append({
                    "collected_at": now_utc_iso(),
                    "symbol": row.get("symbol") or symbol,
                    "funding_time": ms_to_iso(row.get("fundingTime")),
                    "funding_time_ms": int(row.get("fundingTime")),
                    "funding_rate": _coerce_float(row.get("fundingRate")),
                    "mark_price": _coerce_float(row.get("markPrice")),
                })
            next_start = int(payload[-1].get("fundingTime")) + 1
            if next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(self._sleep_seconds)
        return rows

    def collect_binance_open_interest(
        self,
        symbol: str,
        period: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[dict]:
        rows = []
        start_ms = utc_ms(start_date)
        end_ms = utc_ms(end_date, end_of_day=True)
        while start_ms <= end_ms:
            payload = self.get_json(
                f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 500,
                },
            )
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                rows.append({
                    "collected_at": now_utc_iso(),
                    "symbol": row.get("symbol") or symbol,
                    "period": period,
                    "timestamp": ms_to_iso(row.get("timestamp")),
                    "timestamp_ms": int(row.get("timestamp")),
                    "sum_open_interest": _coerce_float(row.get("sumOpenInterest")),
                    "sum_open_interest_value": _coerce_float(row.get("sumOpenInterestValue")),
                })
            next_start = int(payload[-1].get("timestamp")) + 1
            if next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(self._sleep_seconds)
        return rows

    def collect_binance_long_short(
        self,
        symbol: str,
        period: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> list[dict]:
        rows = []
        start_ms = utc_ms(start_date)
        end_ms = utc_ms(end_date, end_of_day=True)
        while start_ms <= end_ms:
            payload = self.get_json(
                f"{BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 500,
                },
            )
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                rows.append({
                    "collected_at": now_utc_iso(),
                    "symbol": row.get("symbol") or symbol,
                    "period": period,
                    "timestamp": ms_to_iso(row.get("timestamp")),
                    "timestamp_ms": int(row.get("timestamp")),
                    "long_short_ratio": _coerce_float(row.get("longShortRatio")),
                    "long_account": _coerce_float(row.get("longAccount")),
                    "short_account": _coerce_float(row.get("shortAccount")),
                })
            next_start = int(payload[-1].get("timestamp")) + 1
            if next_start <= start_ms:
                break
            start_ms = next_start
            time.sleep(self._sleep_seconds)
        return rows
