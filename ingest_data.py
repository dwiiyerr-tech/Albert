"""
Albert data-ingestion CLI.

This CLI collects raw public datasets into JSONL files under data/raw/. It is
designed for replay and research; the live/demo trading loop does not read these
files directly yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from config import CITIES
from data_ingestion import DataCollector
from data_ingestion.storage import load_json, parse_date, write_jsonl


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data_manifest.json"


def _selected_cities(names: list[str] | None) -> list[dict]:
    if not names:
        return list(CITIES)
    wanted = {name.lower() for name in names}
    selected = [city for city in CITIES if city["name"].lower() in wanted]
    missing = sorted(wanted - {city["name"].lower() for city in selected})
    if missing:
        raise SystemExit(f"Unknown city name(s): {', '.join(missing)}")
    return selected


def _date_defaults(args) -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    start = parse_date(args.start_date) if args.start_date else today - dt.timedelta(days=30)
    end = parse_date(args.end_date) if args.end_date else today
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date")
    return start, end


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part) or "query"


def _write(path: Path, rows: list[dict], append: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY RUN: would write {len(rows)} rows -> {path}")
        return
    count = write_jsonl(path, rows, append=append)
    print(f"Wrote {count} rows -> {path}")


def list_sources() -> None:
    manifest = load_json(MANIFEST_PATH)
    print(f"Manifest: {MANIFEST_PATH}")
    for source in manifest.get("sources", []):
        auth = source.get("auth", "none")
        print(f"- {source['name']:<36} {source['category']:<18} auth={auth}")


def run_collection(args) -> None:
    collector = DataCollector()
    data_root = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_DATA_ROOT
    start, end = _date_defaults(args)
    cities = _selected_cities(args.city)
    append = bool(args.append)

    def collect_one(source: str) -> None:
        if source == "polymarket-markets":
            rows = collector.collect_polymarket_markets(query=args.query, limit=args.limit)
            _write(data_root / "polymarket" / f"markets_{_safe_name(args.query)}.jsonl", rows, append, args.dry_run)
            return

        if source == "polymarket-orderbooks":
            if not args.token_id:
                raise SystemExit("--token-id is required for polymarket-orderbooks")
            rows = collector.collect_polymarket_orderbooks(args.token_id)
            _write(data_root / "polymarket" / "orderbooks.jsonl", rows, append, args.dry_run)
            return

        if source == "weather-forecast":
            rows = collector.collect_weather_forecast(cities, days=args.days)
            _write(data_root / "weather" / f"forecasts_{dt.date.today().isoformat()}.jsonl", rows, append, args.dry_run)
            return

        if source == "weather-actuals":
            rows = collector.collect_weather_actuals(cities, start, end)
            _write(data_root / "weather" / f"actuals_{start.isoformat()}_{end.isoformat()}.jsonl", rows, append, args.dry_run)
            return

        if source == "metar":
            rows = collector.collect_metar(cities)
            _write(data_root / "weather" / f"metar_{dt.date.today().isoformat()}.jsonl", rows, append, args.dry_run)
            return

        if source == "btc-ohlcv":
            rows = collector.collect_binance_klines(args.symbol, args.interval, start, end)
            out = data_root / "crypto" / f"{args.symbol}_{args.interval}_ohlcv_{start.isoformat()}_{end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        if source == "btc-funding":
            rows = collector.collect_binance_funding(args.symbol, start, end)
            out = data_root / "crypto" / f"{args.symbol}_funding_{start.isoformat()}_{end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        if source == "btc-open-interest":
            rows = collector.collect_binance_open_interest(args.symbol, args.period, start, end)
            out = data_root / "crypto" / f"{args.symbol}_{args.period}_open_interest_{start.isoformat()}_{end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        if source == "btc-long-short":
            rows = collector.collect_binance_long_short(args.symbol, args.period, start, end)
            out = data_root / "crypto" / f"{args.symbol}_{args.period}_long_short_{start.isoformat()}_{end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        raise SystemExit(f"Unsupported source: {source}")

    if args.source == "all-lite":
        for source in (
            "polymarket-markets",
            "weather-forecast",
            "metar",
            "btc-ohlcv",
            "btc-funding",
            "btc-open-interest",
            "btc-long-short",
        ):
            collect_one(source)
        return

    collect_one(args.source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw public datasets for Albert research/replay.")
    parser.add_argument(
        "--source",
        choices=[
            "all-lite",
            "polymarket-markets",
            "polymarket-orderbooks",
            "weather-forecast",
            "weather-actuals",
            "metar",
            "btc-ohlcv",
            "btc-funding",
            "btc-open-interest",
            "btc-long-short",
        ],
        default="all-lite",
        help="Dataset to collect. all-lite pulls only a small recent sample.",
    )
    parser.add_argument("--list-sources", action="store_true", help="Print the full data manifest and exit.")
    parser.add_argument("--output-dir", help="Override output root (default: Albert/data/raw).")
    parser.add_argument("--append", action="store_true", help="Append instead of overwriting the target JSONL file.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and count rows, but do not write files.")
    parser.add_argument("--start-date", help="Historical start date, YYYY-MM-DD. Default: 30 days ago.")
    parser.add_argument("--end-date", help="Historical end date, YYYY-MM-DD. Default: today.")
    parser.add_argument("--city", action="append", help="Limit weather collection to a configured city. Repeatable.")
    parser.add_argument("--days", type=int, default=7, help="Forecast days for weather-forecast. Default: 7.")
    parser.add_argument("--query", default="temperature", help="Polymarket public-search query. Default: temperature.")
    parser.add_argument("--limit", type=int, default=100, help="Polymarket public-search limit. Default: 100.")
    parser.add_argument("--token-id", action="append", help="CLOB token id for orderbook collection. Repeatable.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol. Default: BTCUSDT.")
    parser.add_argument("--interval", default="1d", help="Binance kline interval. Default: 1d.")
    parser.add_argument("--period", default="1d", help="Binance futures stats period. Default: 1d.")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    run_collection(args)


if __name__ == "__main__":
    main()
