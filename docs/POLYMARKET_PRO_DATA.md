# Polymarket Pro Data Plan

This note lists the data Albert needs to move from a weather-focused demo agent
to a stronger Polymarket research/trading agent. Prioritize official Polymarket
APIs first, then add on-chain analytics providers and category-specific external
truth sources.

## Current Coverage

Albert already tracks:

- Gamma public search for market discovery and token IDs
- CLOB order book snapshots for executable spread/depth/slippage
- Polymarket resolution metadata
- Weather, BTC, and macro features for a subset of market categories

This branch adds manifest entries and collector/CLI hooks for:

- CLOB price history
- Data API trades
- Data API holders
- Data API open interest
- Data API user activity

## Highest-Impact Additions

### 1. Trade Tape

Use:

- `https://data-api.polymarket.com/trades`
- CLOB `getMarketTradesEvents(conditionID)` SDK method where available
- CLOB market websocket `last_trade_price` events for realtime tape

Why it matters:

- Detect aggressive buying/selling before top-of-book fully updates
- Estimate order-flow imbalance and whale prints
- Build wallet-level skill profiles
- Separate real flow from passive spoof-like depth

Features:

- buy/sell volume imbalance over 1m, 5m, 30m, 24h
- large-trade z-score by market liquidity
- post-trade drift after large prints
- wallet win-rate and category specialization once outcomes resolve

### 2. Price History

Use:

- `https://clob.polymarket.com/prices-history`

Why it matters:

- Current Albert sees current price but not path dependency
- A 52% market drifting from 20% is different from a 52% market drifting from 80%

Features:

- 1h/6h/1d momentum
- realized volatility
- max drawdown and rebound
- time-to-resolution price decay
- market-price calibration vs final outcome by category

### 3. Holder and Open-Interest Data

Use:

- `https://data-api.polymarket.com/holders`
- `https://data-api.polymarket.com/oi`
- `https://data-api.polymarket.com/positions` for targeted wallets
- `https://data-api.polymarket.com/activity` for targeted wallets

Why it matters:

- Markets with concentrated holders have different exit risk
- Growing OI with narrowing spread can signal real attention
- Top-holder changes can reveal informed repositioning

Features:

- top-holder concentration ratio
- whale net-flow by outcome
- OI growth over 5m/1h/24h
- top-wallet realized skill by tag/category

### 4. Realtime WebSocket Stream

Use:

- Polymarket CLOB market websocket
- User websocket only for authenticated self-monitoring

Why it matters:

- REST snapshots are enough for slow research, not for execution quality
- Realtime stream helps avoid stale books and late fills

Features:

- live best bid/ask cache
- quote update rate
- stale-book detector
- sudden spread widening alert
- live stop-loss and trailing-stop monitor

### 5. Gamma Metadata, Tags, Series, and Comments

Use:

- `https://gamma-api.polymarket.com/events`
- `https://gamma-api.polymarket.com/markets`
- `https://gamma-api.polymarket.com/tags`
- `https://gamma-api.polymarket.com/series`
- `https://gamma-api.polymarket.com/comments`

Why it matters:

- Tags and series create category priors and correlated exposure groups
- Comments provide event-specific narratives for LLM debate context

Features:

- tag-level Brier calibration
- series-level exposure cap
- same-event correlated market detection
- comment sentiment/catalyst summaries

### 6. On-Chain Audit Layer

Use:

- Goldsky pipelines
- Dune SQL
- Allium
- CryptoHouse/ClickHouse

Why it matters:

- Polymarket CLOB matching is offchain but settlement and token flows land on
  Polygon. On-chain data is the independent audit layer for fills, balances,
  splits, merges, and redeems.

Features:

- split/merge/redeem tracking
- wallet balance changes
- settlement reconciliation
- self-trade/wash-risk heuristics
- market maker inventory estimation

## External Truth Sources by Category

Albert should not rely on Polymarket data alone. A pro agent needs independent
models for the underlying event.

Weather:

- Open-Meteo forecast/archive
- Aviation Weather METAR
- NOAA/NWS/NHC feeds for US severe weather and hurricanes

Crypto:

- exchange OHLCV, funding, open interest, liquidations, options volatility
- ETF flow and stablecoin liquidity data where available

Sports:

- official league schedules/injuries
- sportsbook odds as a market-implied baseline
- play-by-play and team strength features

Politics/macro:

- official election boards, FEC, polling aggregators where licensed
- FRED, BLS, BEA, Federal Reserve calendars and releases
- major-event calendars with release timestamps

## Implementation Priority

1. Backfill CLOB `prices-history` for active token IDs.
2. Collect Data API `trades`, `holders`, `oi`, and selected wallet `activity`.
3. Add feature builder for flow and microstructure features.
4. Feed those features into EV calculation as risk/edge adjustments.
5. Add websocket cache for live/demo execution.
6. Add wallet profiler and category calibration dashboards.

Example local commands:

```bash
python ingest_data.py --source polymarket-price-history --token-id TOKEN_ID --start-date 2026-05-01 --end-date 2026-05-17
python ingest_data.py --source polymarket-trades --condition-id CONDITION_ID --limit 500
python ingest_data.py --source polymarket-holders --condition-id CONDITION_ID --limit 20
python ingest_data.py --source polymarket-open-interest --condition-id CONDITION_ID
python ingest_data.py --source polymarket-user-activity --user 0xPROFILE_WALLET --condition-id CONDITION_ID
```

## Source Notes

- Polymarket API overview: https://docs.polymarket.com/api-reference/introduction
- Market-data overview: https://docs.polymarket.com/market-data/overview
- CLOB public methods: https://docs.polymarket.com/trading/clients/public
- Data API trades: https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets
- Data API holders: https://docs.polymarket.com/api-reference/core/get-top-holders-for-markets
- Data API positions/activity: https://docs.polymarket.com/api-reference/core/get-current-positions-for-a-user
- Rate limits: https://docs.polymarket.com/api-reference/rate-limits
- On-chain data resources: https://docs.polymarket.com/resources/blockchain-data
