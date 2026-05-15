# MiroWeather

**A combined AI agent born from [MiroFish](https://github.com/666ghj/MiroFish) and [WeatherBot](https://github.com/alteregoeth-ai/weatherbot).**

---

## What is MiroWeather?

MiroWeather fuses two open-source projects into a single intelligent agent:

| Origin | Contribution |
|--------|-------------|
| **MiroFish** (666ghj) | Multi-agent simulation engine: analyst personas, debate rounds, consensus probability, knowledge graph, ReportAgent with tool use |
| **WeatherBot** (alteregoeth-ai) | Real weather data (ECMWF, GFS, METAR), Polymarket scanner, Expected Value formula, Kelly Criterion sizing, stop-loss / trailing-stop |

Instead of feeding raw forecast temperatures directly to a trading algorithm, MiroWeather runs a **structured multi-agent debate**: four AI weather analysts with distinct personalities argue over the forecast, reconcile model disagreements, and produce a calibrated consensus probability — which then drives trading decisions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MiroWeather Agent                        │
│                                                                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────┐ │
│  │ WeatherData  │───▶│  WeatherSimulation │───▶│  EV / Kelly │ │
│  │  Layer       │    │  (MiroFish agents) │    │  Calculator │ │
│  │  ECMWF / GFS │    │  4 analysts        │    │             │ │
│  │  METAR obs   │    │  3 debate rounds   │    │  MIN_EV=0.10│ │
│  └──────────────┘    │  Claude API        │    └──────┬──────┘ │
│                      └────────┬───────────┘           │        │
│  ┌──────────────┐             │                        ▼        │
│  │  Knowledge   │◀────────────┘            ┌─────────────────┐ │
│  │  Graph       │  city climatology        │ Position Manager│ │
│  │  model MAE   │  model accuracy          │ stop-loss       │ │
│  └──────────────┘                          │ trailing-stop   │ │
│                      ┌────────────────┐    └─────────────────┘ │
│  ┌──────────────┐    │ Report         │                         │
│  │  Polymarket  │    │ Generator      │                         │
│  │  Scanner     │    │ (tool use)     │                         │
│  └──────────────┘    └────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (Anthropic native)
export LLM_PROVIDER="anthropic"
export LLM_API_KEY="sk-ant-..."
export LLM_MODEL="claude-sonnet-4-6"
export VISUAL_CROSSING_API_KEY="..."   # optional, for historical validation

# Or use any OpenAI-compatible API:
# export LLM_PROVIDER="openai-compatible"
# export LLM_API_KEY="..."
# export LLM_MODEL="gpt-4.1"
# export LLM_BASE_URL="https://api.openai.com/v1"

# 3. Run one analysis cycle (dry run — no real trades)
python main.py --run

# 4. Run in daemon mode (cycles every 60 minutes)
python main.py --daemon

# 5. Show position summary
python main.py --positions

# 6. Target a specific day ahead
python main.py --run --days-ahead 2

# 7. Live trading mode (requires POLYMARKET_API_KEY)
python main.py --run --live
```

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_EV` | 0.10 | Minimum expected value to enter a trade |
| `MAX_SPREAD` | 0.03 | Maximum bid-ask spread |
| `MIN_VOLUME` | 500 | Minimum contract volume |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |
| `MAX_TRADE_SIZE_USD` | $20 | Maximum position size |
| `SIM_ROUNDS` | 3 | Debate rounds per city per bucket |
| `MAX_AGENTS_PER_SIM` | 4 | Number of analyst personas |
| `CONSENSUS_THRESHOLD` | 0.65 | High-confidence signal threshold |
| `STOP_LOSS_PCT` | 0.20 | Stop-loss at 20% below entry |

---

## Agent Personas

| Analyst | Bias | Style |
|---------|------|-------|
| **Alex (ECMWF Purist)** | Trusts ECMWF over GFS | Data-driven, references ensemble spread |
| **Jordan (Mesoscale Expert)** | Prioritises METAR + urban heat islands | Detailed, micro-climate focus |
| **Sam (Climatologist)** | Weights historical anomalies | Cautious, references 30-year normals |
| **Casey (Contrarian)** | Questions model consensus | Provocative, hunts tail risks |

### MiroFish-Style Learning Extensions

- Full debate transcripts are persisted to memory for later audit and self-play.
- Persona estimates are stored by name so resolved markets can score each analyst.
- Future votes are weighted by historical persona Brier score when enough outcomes exist.
- Dynamic personas can be created by self-play reflection when a missing viewpoint is detected.
- Multi-provider LLM ensemble is supported with `LLM_ENSEMBLE`, rotating personas across configured providers/models.

Example ensemble:

```bash
export LLM_ENSEMBLE="anthropic:claude-sonnet-4-6,openai-compatible:gpt-4.1:https://api.openai.com/v1"
```

---

## Data Sources

- **Open-Meteo** — ECMWF IFS 0.25° and GFS seamless forecasts (free)
- **Aviation Weather (NOAA)** — Live METAR temperature observations (free)
- **Visual Crossing** — Historical temperature validation (free tier)
- **Polymarket** — Prediction market prices and order book data

---

## Project Structure

```
Albert/
├── main.py                    # Orchestrator + CLI entry point
├── config.py                  # All parameters and city list
├── weather_data.py            # ECMWF / GFS / METAR data fetching
├── requirements.txt
├── simulation/
│   ├── agents.py              # Multi-agent debate (MiroFish-inspired)
│   ├── knowledge_graph.py     # City climatology + model accuracy graph
│   └── report_generator.py    # Claude-powered structured reports
└── trading/
    ├── ev_calculator.py       # EV + Kelly sizing (WeatherBot-inspired)
    ├── market_scanner.py      # Polymarket weather market scanner
    └── position_manager.py    # Paper trade tracking with stops
```
