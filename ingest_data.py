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
from data_ingestion.collectors import select_polymarket_targets
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


def _latest_30_day_window(start: dt.date, end: dt.date, source: str) -> tuple[dt.date, dt.date]:
    cutoff = end - dt.timedelta(days=30)
    if start < cutoff:
        print(
            f"NOTE: {source} is limited by Binance to the latest 30 days. "
            f"Using {cutoff.isoformat()}..{end.isoformat()}."
        )
        start = cutoff
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


def _write_bundle(
    data_root: Path,
    query: str,
    start: dt.date,
    end: dt.date,
    rows_by_name: dict[str, list[dict]],
    append: bool,
    dry_run: bool,
) -> None:
    suffix = f"{start.isoformat()}_{end.isoformat()}"
    paths = {
        "markets": data_root / "polymarket" / f"markets_{_safe_name(query)}.jsonl",
        "orderbooks": data_root / "polymarket" / "orderbooks.jsonl",
        "price_history": data_root / "polymarket" / f"price_history_{suffix}.jsonl",
        "trades": data_root / "polymarket" / "data_api_trades.jsonl",
        "holders": data_root / "polymarket" / "holders.jsonl",
        "open_interest": data_root / "polymarket" / "open_interest.jsonl",
        "user_activity": data_root / "polymarket" / "user_activity.jsonl",
    }
    for name, rows in rows_by_name.items():
        _write(paths[name], rows, append, dry_run)


def list_sources() -> None:
    manifest = load_json(MANIFEST_PATH)
    print(f"Manifest: {MANIFEST_PATH}")
    print("- polymarket-pro-refresh             composite          auth=none")
    for source in manifest.get("sources", []):
        auth = source.get("auth", "none")
        print(f"- {source['name']:<36} {source['category']:<18} auth={auth}")


def run_collection(args) -> None:
    collector = DataCollector(sleep_seconds=args.request_sleep)
    data_root = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_DATA_ROOT
    start, end = _date_defaults(args)
    cities = _selected_cities(args.city)
    append = bool(args.append)

    def collect_one(source: str) -> None:
        if source == "polymarket-pro-refresh":
            market_rows = collector.collect_polymarket_markets(query=args.query, limit=args.limit)
            targets = select_polymarket_targets(market_rows, max_markets=args.max_markets)
            token_ids = targets["token_ids"]
            condition_ids = targets["condition_ids"]
            print(
                "Polymarket pro refresh targets: "
                f"{len(targets['markets'])} markets, {len(condition_ids)} conditions, {len(token_ids)} tokens"
            )

            rows_by_name = {
                "markets": market_rows,
                "orderbooks": collector.collect_polymarket_orderbooks(token_ids),
                "price_history": collector.collect_polymarket_price_history(
                    token_ids,
                    start,
                    end,
                    interval=args.history_interval,
                    fidelity=args.fidelity,
                ),
                "trades": collector.collect_polymarket_trades(condition_ids, limit=args.limit),
                "holders": collector.collect_polymarket_holders(
                    condition_ids,
                    limit=args.limit,
                    min_balance=args.min_balance,
                ),
                "open_interest": collector.collect_polymarket_open_interest(condition_ids),
            }
            if args.user:
                rows_by_name["user_activity"] = collector.collect_polymarket_user_activity(
                    args.user,
                    condition_ids,
                    limit=args.limit,
                )
            _write_bundle(data_root, args.query, start, end, rows_by_name, append, args.dry_run)
            return

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

        if source == "polymarket-price-history":
            if not args.token_id:
                raise SystemExit("--token-id is required for polymarket-price-history")
            rows = collector.collect_polymarket_price_history(
                args.token_id,
                start,
                end,
                interval=args.history_interval,
                fidelity=args.fidelity,
            )
            out = data_root / "polymarket" / f"price_history_{start.isoformat()}_{end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        if source == "polymarket-trades":
            rows = collector.collect_polymarket_trades(args.condition_id, limit=args.limit)
            _write(data_root / "polymarket" / "data_api_trades.jsonl", rows, append, args.dry_run)
            return

        if source == "polymarket-holders":
            if not args.condition_id:
                raise SystemExit("--condition-id is required for polymarket-holders")
            rows = collector.collect_polymarket_holders(
                args.condition_id,
                limit=args.limit,
                min_balance=args.min_balance,
            )
            _write(data_root / "polymarket" / "holders.jsonl", rows, append, args.dry_run)
            return

        if source == "polymarket-open-interest":
            rows = collector.collect_polymarket_open_interest(args.condition_id)
            _write(data_root / "polymarket" / "open_interest.jsonl", rows, append, args.dry_run)
            return

        if source == "polymarket-user-activity":
            if not args.user:
                raise SystemExit("--user is required for polymarket-user-activity")
            rows = collector.collect_polymarket_user_activity(args.user, args.condition_id, limit=args.limit)
            _write(data_root / "polymarket" / "user_activity.jsonl", rows, append, args.dry_run)
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
            effective_start, effective_end = _latest_30_day_window(start, end, source)
            rows = collector.collect_binance_open_interest(args.symbol, args.period, effective_start, effective_end)
            out = data_root / "crypto" / f"{args.symbol}_{args.period}_open_interest_{effective_start.isoformat()}_{effective_end.isoformat()}.jsonl"
            _write(out, rows, append, args.dry_run)
            return

        if source == "btc-long-short":
            effective_start, effective_end = _latest_30_day_window(start, end, source)
            rows = collector.collect_binance_long_short(args.symbol, args.period, effective_start, effective_end)
            out = data_root / "crypto" / f"{args.symbol}_{args.period}_long_short_{effective_start.isoformat()}_{effective_end.isoformat()}.jsonl"
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
            "polymarket-pro-refresh",
            "polymarket-markets",
            "polymarket-orderbooks",
            "polymarket-price-history",
            "polymarket-trades",
            "polymarket-holders",
            "polymarket-open-interest",
            "polymarket-user-activity",
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
    parser.add_argument("--max-markets", type=int, default=20,
                        help="Max active markets to enrich for polymarket-pro-refresh. Default: 20.")
    parser.add_argument("--token-id", action="append", help="CLOB token id for orderbook collection. Repeatable.")
    parser.add_argument("--condition-id", action="append", help="Polymarket condition id. Repeatable.")
    parser.add_argument("--user", help="Polymarket user/profile wallet address for user activity.")
    parser.add_argument("--min-balance", type=int, default=1, help="Minimum holder balance for holders. Default: 1.")
    parser.add_argument("--history-interval", default="1d",
                        help="Polymarket price-history interval: max, 1w, 1d, 6h, or 1h. Default: 1d.")
    parser.add_argument("--fidelity", type=int, help="Optional Polymarket price-history fidelity/resolution.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol. Default: BTCUSDT.")
    parser.add_argument("--interval", default="1d", help="Binance kline interval. Default: 1d.")
    parser.add_argument("--period", default="1d", help="Binance futures stats period. Default: 1d.")
    parser.add_argument("--request-sleep", type=float, default=0.3,
                        help="Seconds to sleep between paginated/provider requests. Default: 0.3.")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    run_collection(args)


if __name__ == "__main__":
    main()
