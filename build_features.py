"""Build processed research features from Albert raw JSONL backfills."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_ingestion.features import (
    build_btc_risk_regimes,
    build_weather_monthly_normals,
    latest_path,
    read_jsonl,
)
from data_ingestion.storage import write_jsonl


REPO_ROOT = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


def _require(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"Missing required input for {label}")
    return path


def build_btc(args) -> None:
    crypto_root = Path(args.raw_dir).resolve() / "crypto"
    out_dir = Path(args.output_dir).resolve()
    ohlcv_path = Path(args.ohlcv).resolve() if args.ohlcv else latest_path(crypto_root, "BTCUSDT_1d_ohlcv_*.jsonl")
    funding_path = Path(args.funding).resolve() if args.funding else latest_path(crypto_root, "BTCUSDT_funding_*.jsonl")
    oi_path = Path(args.open_interest).resolve() if args.open_interest else latest_path(crypto_root, "BTCUSDT_1d_open_interest_*.jsonl")
    ls_path = Path(args.long_short).resolve() if args.long_short else latest_path(crypto_root, "BTCUSDT_1d_long_short_*.jsonl")

    rows = build_btc_risk_regimes(
        ohlcv_rows=read_jsonl(_require(ohlcv_path, "BTC OHLCV")),
        funding_rows=read_jsonl(funding_path) if funding_path else [],
        open_interest_rows=read_jsonl(oi_path) if oi_path else [],
        long_short_rows=read_jsonl(ls_path) if ls_path else [],
    )
    out = out_dir / "btc_risk_regimes_1d.jsonl"
    count = write_jsonl(out, rows)
    print(f"Wrote {count} rows -> {out}")


def build_weather(args) -> None:
    weather_root = Path(args.raw_dir).resolve() / "weather"
    out_dir = Path(args.output_dir).resolve()
    actuals_path = Path(args.weather_actuals).resolve() if args.weather_actuals else latest_path(weather_root, "actuals_*.jsonl")
    rows = build_weather_monthly_normals(read_jsonl(_require(actuals_path, "weather actuals")))
    out = out_dir / "weather_city_month_normals.jsonl"
    count = write_jsonl(out, rows)
    print(f"Wrote {count} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed Albert research features.")
    parser.add_argument("--source", choices=["all", "btc-regimes", "weather-normals"], default="all")
    parser.add_argument("--raw-dir", default=str(RAW_ROOT), help="Raw data root. Default: Albert/data/raw")
    parser.add_argument("--output-dir", default=str(PROCESSED_ROOT), help="Processed output root. Default: Albert/data/processed")
    parser.add_argument("--ohlcv", help="Explicit BTC OHLCV JSONL path.")
    parser.add_argument("--funding", help="Explicit BTC funding JSONL path.")
    parser.add_argument("--open-interest", help="Explicit BTC open-interest JSONL path.")
    parser.add_argument("--long-short", help="Explicit BTC long/short JSONL path.")
    parser.add_argument("--weather-actuals", help="Explicit weather actuals JSONL path.")
    args = parser.parse_args()

    if args.source in {"all", "btc-regimes"}:
        build_btc(args)
    if args.source in {"all", "weather-normals"}:
        build_weather(args)


if __name__ == "__main__":
    main()
