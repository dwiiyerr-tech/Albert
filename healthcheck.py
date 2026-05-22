"""
Albert healthcheck.

Runs fast operational checks for demo/daemon readiness:
  - config import and live-trading lock state
  - local state files can be loaded
  - configured LLM can answer a tiny prompt
  - core external APIs respond with parseable JSON
  - Telegram bot token can be verified without sending a message
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

import config
from llm_client import LLMClient
from main import live_trading_block_reason
from learning import ExperienceMemory
from simulation.knowledge_graph import WeatherKnowledgeGraph
from trading import PositionManager


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    latency_ms: int = 0


def _timed(name: str, fn) -> CheckResult:
    started = time.monotonic()
    try:
        detail = fn()
        status = "ok"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        status = "fail"
    return CheckResult(
        name=name,
        status=status,
        detail=str(detail),
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def _warn(name: str, detail: str, latency_ms: int = 0) -> CheckResult:
    return CheckResult(name=name, status="warn", detail=detail, latency_ms=latency_ms)


def _http_json(
    name: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 10,
    optional: bool = False,
) -> CheckResult:
    started = time.monotonic()
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, (dict, list)):
            raise RuntimeError(f"unexpected JSON type {type(payload).__name__}")
        size_hint = len(payload) if isinstance(payload, list) else len(payload.keys())
        return CheckResult(
            name=name,
            status="ok",
            detail=f"HTTP {resp.status_code}, json={type(payload).__name__}, size={size_hint}",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        result = CheckResult(
            name=name,
            status="warn" if optional else "fail",
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return result


def check_config() -> CheckResult:
    def _run() -> str:
        provider = config.LLM_PROVIDER
        if provider not in {"anthropic", "openai", "openai-compatible", "compatible", "mock"}:
            raise RuntimeError(f"unsupported LLM_PROVIDER={provider!r}")
        mode = config.DEFAULT_MODE
        if mode not in {"dry", "demo", "live"}:
            raise RuntimeError(f"unsupported DEFAULT_MODE={mode!r}")
        return f"provider={provider}, mode={mode}, live_lock={live_trading_block_reason() or 'open'}"

    return _timed("config", _run)


def check_state_files() -> CheckResult:
    def _run() -> str:
        memory = ExperienceMemory()
        graph = WeatherKnowledgeGraph()
        positions = PositionManager()
        return (
            f"memory_predictions={len(memory.predictions)}, "
            f"cities={len(graph.cities)}, "
            f"open_positions={len(positions.open_positions)}"
        )

    return _timed("state-files", _run)


def check_llm() -> CheckResult:
    def _run() -> str:
        client = LLMClient()
        text = client.text(
            system="Return the word ok.",
            messages=[{"role": "user", "content": "Healthcheck"}],
            max_tokens=16,
        )
        if not text:
            raise RuntimeError("empty LLM response")
        return f"provider={client.provider}, model={client.model}, chars={len(text)}"

    return _timed("llm", _run)


def check_network() -> list[CheckResult]:
    return [
        _http_json(
            "polymarket-gamma",
            f"{config.POLYMARKET_GAMMA}/public-search",
            params={"q": "temperature", "active": "true", "closed": "false", "limit": 1},
        ),
        _http_json(
            "polymarket-clob",
            f"{config.POLYMARKET_BASE}/markets",
            params={"limit": 1},
        ),
        _http_json(
            "open-meteo",
            f"{config.OPEN_METEO_BASE}/forecast",
            params={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "daily": "temperature_2m_max",
                "forecast_days": 1,
                "temperature_unit": "celsius",
            },
        ),
        _http_json(
            "aviation-weather",
            f"{config.AVIATION_WEATHER_BASE}/metar",
            params={"ids": "KJFK", "format": "json"},
        ),
        _http_json(
            "binance-spot",
            "https://api.binance.com/api/v3/time",
            optional=True,
        ),
        check_telegram(),
    ]


def check_telegram() -> CheckResult:
    if config.REMOTE_CONTROL_PROVIDER != "telegram":
        return _warn("telegram", f"provider={config.REMOTE_CONTROL_PROVIDER!r}; skipped")
    if not config.TELEGRAM_BOT_TOKEN:
        status = "fail" if config.REMOTE_CONTROL_ENABLED else "warn"
        return CheckResult(
            name="telegram",
            status=status,
            detail="TELEGRAM_BOT_TOKEN is not configured; getMe skipped",
        )

    started = time.monotonic()
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getMe",
            timeout=10,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            payload = resp.json()
        except Exception as exc:
            return CheckResult(
                name="telegram",
                status="fail",
                detail=f"HTTP {resp.status_code}, invalid JSON: {type(exc).__name__}",
                latency_ms=latency_ms,
            )
        if not resp.ok or not payload.get("ok"):
            description = payload.get("description") or "Telegram getMe failed"
            return CheckResult(
                name="telegram",
                status="fail",
                detail=f"HTTP {resp.status_code}, {description}",
                latency_ms=latency_ms,
            )
        bot = payload.get("result") or {}
        username = bot.get("username") or bot.get("first_name") or "unknown"
        allowed = len([x for x in config.REMOTE_ALLOWED_CHAT_IDS.split(",") if x.strip()])
        notifications = len([x for x in config.REMOTE_NOTIFICATION_CHAT_IDS.split(",") if x.strip()])
        return CheckResult(
            name="telegram",
            status="ok",
            detail=f"getMe ok, bot=@{username}, allowed_chats={allowed}, notification_chats={notifications}",
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return CheckResult(
            name="telegram",
            status="fail",
            detail=f"{type(exc).__name__}: Telegram getMe request failed",
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def run_checks(*, skip_network: bool = False) -> list[CheckResult]:
    results = [check_config(), check_state_files(), check_llm()]
    if skip_network:
        results.append(_warn("network", "skipped by --skip-network"))
    else:
        results.extend(check_network())
    return results


def print_table(results: list[CheckResult]) -> None:
    width = max(len(r.name) for r in results)
    for result in results:
        print(
            f"{result.status.upper():<5} "
            f"{result.name:<{width}} "
            f"{result.latency_ms:>5}ms  "
            f"{result.detail}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Albert demo/runtime dependencies.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--skip-network", action="store_true", help="Only check local config/state/LLM")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args(argv)

    results = run_checks(skip_network=args.skip_network)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print_table(results)

    has_fail = any(r.status == "fail" for r in results)
    has_warn = any(r.status == "warn" for r in results)
    return 1 if has_fail or (args.strict and has_warn) else 0


if __name__ == "__main__":
    raise SystemExit(main())
