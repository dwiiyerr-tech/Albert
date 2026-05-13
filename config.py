"""
MiroWeather Configuration
Combined configuration for weather simulation + prediction trading agent.
"""
import os
import sys


def _require_env(name: str) -> str:
    """Return env var value or exit with a clear message if missing."""
    val = os.getenv(name, "")
    if not val:
        print(f"ERROR: required environment variable {name!r} is not set. "
              f"Set it before running MiroWeather.", file=sys.stderr)
        sys.exit(1)
    return val


# ─── LLM / Claude API ────────────────────────────────────────────────────────
# Validated at import time: agent cannot start without this key.
ANTHROPIC_API_KEY = _require_env("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# ─── Weather Data Sources ─────────────────────────────────────────────────────
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
AVIATION_WEATHER_BASE = "https://aviationweather.gov/api/data"
# Optional — agent falls back to Open-Meteo archive if not set
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")

# ─── Polymarket ───────────────────────────────────────────────────────────────
POLYMARKET_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")

# ─── Trading Parameters ───────────────────────────────────────────────────────
MIN_EV = 0.10                # minimum expected value to enter a trade
MAX_SPREAD = 0.03            # maximum bid-ask spread (3%)
MIN_VOLUME = 500             # minimum contract volume
MIN_HOURS_TO_RESOLUTION = 2
MAX_HOURS_TO_RESOLUTION = 72
KELLY_FRACTION = 0.25        # fractional Kelly (25%)
MAX_TRADE_SIZE_USD = 20.0
STOP_LOSS_PCT = 0.20         # 20% stop-loss below entry
TRAILING_STOP_TRIGGER = 0.20 # activate trailing stop after 20% profit

# ─── Simulation Parameters ────────────────────────────────────────────────────
SIM_ROUNDS = 3               # debate rounds per city simulation (classic mode)
MAX_AGENTS_PER_SIM = 4       # weather analyst agents per city
CONSENSUS_THRESHOLD = 0.65   # probability threshold for high-confidence signal
HIGH_SPREAD_THRESHOLD_F = 8.0  # model disagreement above this → force low-confidence

# ─── Scenario Speculation ─────────────────────────────────────────────────────
# When enabled, simulation uses a 3-phase protocol:
#   Phase 1 — Morgan generates 2-3 explicit future weather scenarios
#   Phase 2 — 4 analysts estimate P(YES | each scenario) independently
#   Phase 3 — River synthesises: P(YES) = Σ P(scenario_i) × mean_P(YES|scenario_i)
# This is more accurate than naive averaging, especially when models disagree.
SCENARIO_SPECULATION = os.getenv("SCENARIO_SPECULATION", "true").lower() != "false"
# Only activate scenario mode when model spread exceeds this threshold (°F).
# Set to 0.0 to always use scenarios; set to 99.0 to always use classic debate.
SCENARIO_THRESHOLD_F = float(os.getenv("SCENARIO_THRESHOLD_F", "0.0"))

# ─── Monitored Cities ────────────────────────────────────────────────────────
CITIES = [
    {"name": "New York",     "lat": 40.7128,  "lon": -74.0060,  "metar": "KJFK", "tz": "America/New_York"},
    {"name": "Chicago",      "lat": 41.8781,  "lon": -87.6298,  "metar": "KORD", "tz": "America/Chicago"},
    {"name": "Miami",        "lat": 25.7617,  "lon": -80.1918,  "metar": "KMIA", "tz": "America/New_York"},
    {"name": "Dallas",       "lat": 32.7767,  "lon": -96.7970,  "metar": "KDFW", "tz": "America/Chicago"},
    {"name": "Seattle",      "lat": 47.6062,  "lon": -122.3321, "metar": "KSEA", "tz": "America/Los_Angeles"},
    {"name": "Atlanta",      "lat": 33.7490,  "lon": -84.3880,  "metar": "KATL", "tz": "America/New_York"},
    {"name": "London",       "lat": 51.5074,  "lon": -0.1278,   "metar": "EGLL", "tz": "Europe/London"},
    {"name": "Paris",        "lat": 48.8566,  "lon": 2.3522,    "metar": "LFPG", "tz": "Europe/Paris"},
    {"name": "Tokyo",        "lat": 35.6762,  "lon": 139.6503,  "metar": "RJTT", "tz": "Asia/Tokyo"},
    {"name": "Sydney",       "lat": -33.8688, "lon": 151.2093,  "metar": "YSSY", "tz": "Australia/Sydney"},
    {"name": "Toronto",      "lat": 43.6532,  "lon": -79.3832,  "metar": "CYYZ", "tz": "America/Toronto"},
    {"name": "Los Angeles",  "lat": 34.0522,  "lon": -118.2437, "metar": "KLAX", "tz": "America/Los_Angeles"},
    {"name": "São Paulo",    "lat": -23.5505, "lon": -46.6333,  "metar": "SBGR", "tz": "America/Sao_Paulo"},
    {"name": "Mumbai",       "lat": 19.0760,  "lon": 72.8777,   "metar": "VABB", "tz": "Asia/Kolkata"},
    {"name": "Dubai",        "lat": 25.2048,  "lon": 55.2708,   "metar": "OMDB", "tz": "Asia/Dubai"},
    {"name": "Singapore",    "lat": 1.3521,   "lon": 103.8198,  "metar": "WSSS", "tz": "Asia/Singapore"},
    {"name": "Berlin",       "lat": 52.5200,  "lon": 13.4050,   "metar": "EDDB", "tz": "Europe/Berlin"},
    {"name": "Moscow",       "lat": 55.7558,  "lon": 37.6173,   "metar": "UUEE", "tz": "Europe/Moscow"},
    {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816,  "metar": "SAEZ", "tz": "America/Argentina/Buenos_Aires"},
    {"name": "Cairo",        "lat": 30.0444,  "lon": 31.2357,   "metar": "HECA", "tz": "Africa/Cairo"},
]

# ─── Simulation ──────────────────────────────────────────────────────────────
SIM_STATE_FILE = "sim_state.json"
UPDATE_INTERVAL_SECONDS = 3600  # 60 minutes
