"""
MiroWeather Configuration
Combined configuration for weather simulation + prediction trading agent.
"""
import os
import sys


def _load_dotenv() -> None:
    """Load .env file into os.environ without overriding already-set env vars."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


def _require_env(name: str) -> str:
    """Return env var value or exit with a clear message if missing."""
    val = os.getenv(name, "")
    if not val:
        print(f"ERROR: required environment variable {name!r} is not set. "
              f"Run 'python main.py --setup' to configure Albert.", file=sys.stderr)
        sys.exit(1)
    return val


def _first_env(*names: str) -> str:
    for name in names:
        val = os.getenv(name, "")
        if val:
            return val
    return ""


def require_llm_api_key() -> str:
    """Return the configured LLM API key, validating only when a client is needed."""
    if os.getenv("LLM_PROVIDER", "anthropic").lower() == "mock":
        return "mock"
    val = _first_env("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    if not val:
        print(
            "ERROR: no LLM API key is configured. Set LLM_API_KEY, "
            "ANTHROPIC_API_KEY, or OPENAI_API_KEY, or run 'python main.py --setup'.",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


def require_anthropic_api_key() -> str:
    """Backward-compatible alias for older imports."""
    return require_llm_api_key()


# ─── LLM API ─────────────────────────────────────────────────────────────────
# Generic LLM settings. Provider values:
#   anthropic          Native Anthropic Messages API
#   openai-compatible  OpenAI-compatible /chat/completions APIs
#   mock               Deterministic offline responses for smoke tests only
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_API_KEY = _first_env("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("ANTHROPIC_BASE_URL", ""))
# Optional comma-separated ensemble, e.g.
#   mock,openai-compatible:gpt-4.1:https://api.openai.com/v1
LLM_ENSEMBLE = os.getenv("LLM_ENSEMBLE", "").strip()

# Backward-compatible Anthropic names.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = LLM_MODEL
# Optional: custom base URL for API-compatible proxies or private deployments.
# Leave blank to use the default Anthropic endpoint.
ANTHROPIC_BASE_URL: str = LLM_BASE_URL

# ─── Weather Data Sources ─────────────────────────────────────────────────────
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
AVIATION_WEATHER_BASE = "https://aviationweather.gov/api/data"
# Optional — agent falls back to Open-Meteo archive if not set
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")

# ─── Polymarket ───────────────────────────────────────────────────────────────
POLYMARKET_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
# Ethereum private key for signing CLOB orders (required for live trading).
# Format: hex string with or without 0x prefix.
POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
# Optional proxy wallet address (used when trading via a Polymarket proxy contract).
POLYMARKET_PROXY_ADDRESS: str = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

# ─── Runtime Mode ────────────────────────────────────────────────────────────
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "dry").lower()
DEMO_POSITIONS_FILE = os.getenv("DEMO_POSITIONS_FILE", "positions_demo.json")

# ─── Remote Control / Telegram Bot ───────────────────────────────────────────
REMOTE_CONTROL_ENABLED = os.getenv("REMOTE_CONTROL_ENABLED", "false").lower() == "true"
REMOTE_CONTROL_PROVIDER = os.getenv("REMOTE_CONTROL_PROVIDER", "telegram").lower()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
REMOTE_ALLOWED_CHAT_IDS = os.getenv("REMOTE_ALLOWED_CHAT_IDS", "")
REMOTE_ALLOWED_COMMANDS = os.getenv(
    "REMOTE_ALLOWED_COMMANDS",
    "status,positions,signals,learning,pnl,pause,resume,dry_run_once,demo_once",
)
REMOTE_ALLOW_LIVE = os.getenv("REMOTE_ALLOW_LIVE", "false").lower() == "true"
REMOTE_AUDIT_LOG = os.getenv("REMOTE_AUDIT_LOG", "remote_control.log")
REMOTE_POLL_INTERVAL_SECONDS = float(os.getenv("REMOTE_POLL_INTERVAL_SECONDS", "2.0"))
REMOTE_NOTIFICATION_CHAT_IDS = os.getenv("REMOTE_NOTIFICATION_CHAT_IDS", "")
REMOTE_NOTIFY_CYCLE_SUMMARY = os.getenv("REMOTE_NOTIFY_CYCLE_SUMMARY", "true").lower() != "false"
REMOTE_NOTIFY_ERRORS = os.getenv("REMOTE_NOTIFY_ERRORS", "true").lower() != "false"
REMOTE_DAILY_PNL_ENABLED = os.getenv("REMOTE_DAILY_PNL_ENABLED", "true").lower() != "false"
REMOTE_PNL_REPORT_INTERVAL_HOURS = float(os.getenv("REMOTE_PNL_REPORT_INTERVAL_HOURS", "24.0"))
REMOTE_PNL_REPORT_ON_START = os.getenv("REMOTE_PNL_REPORT_ON_START", "false").lower() == "true"
REMOTE_DEMO_BALANCE = float(os.getenv("REMOTE_DEMO_BALANCE", "1000.0"))
REMOTE_DEMO_POSITIONS_FILE = os.getenv(
    "REMOTE_DEMO_POSITIONS_FILE",
    os.path.join(".demo_runs", "telegram_demo_positions.json"),
)
REMOTE_DEMO_TOKEN_BUDGET = int(os.getenv("REMOTE_DEMO_TOKEN_BUDGET", "200000"))
REMOTE_DEMO_SIM_ROUNDS = int(os.getenv("REMOTE_DEMO_SIM_ROUNDS", "1"))

# ─── Trading Parameters ───────────────────────────────────────────────────────
MIN_EV = float(os.getenv("MIN_EV", "0.10"))
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.03"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "500"))
MIN_ENTRY_PRICE = float(os.getenv("MIN_ENTRY_PRICE", "0.03"))
MAX_ENTRY_PRICE = float(os.getenv("MAX_ENTRY_PRICE", "0.95"))
MIN_PROB_EDGE = float(os.getenv("MIN_PROB_EDGE", "0.05"))
MIN_ORDERBOOK_DEPTH_USD = float(os.getenv("MIN_ORDERBOOK_DEPTH_USD", "1.0"))
MAX_ORDERBOOK_SLIPPAGE = float(os.getenv("MAX_ORDERBOOK_SLIPPAGE", "0.02"))
MAX_POSITIONS_PER_CITY_DATE = int(os.getenv("MAX_POSITIONS_PER_CITY_DATE", "1"))
MAX_EXPOSURE_PER_CITY_DATE_USD = float(os.getenv("MAX_EXPOSURE_PER_CITY_DATE_USD", "2.0"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))
MAX_TOTAL_DEPLOYED_USD = float(os.getenv("MAX_TOTAL_DEPLOYED_USD", "100.0"))
MAX_PORTFOLIO_HEAT_USD = float(os.getenv("MAX_PORTFOLIO_HEAT_USD", "20.0"))
MAX_EXPOSURE_PER_TARGET_DATE_USD = float(os.getenv("MAX_EXPOSURE_PER_TARGET_DATE_USD", "20.0"))
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "25.0"))
MAX_DRAWDOWN_USD = float(os.getenv("MAX_DRAWDOWN_USD", "50.0"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "20"))
MIN_REWARD_RISK_RATIO = float(os.getenv("MIN_REWARD_RISK_RATIO", "1.5"))
MIN_HOURS_TO_RESOLUTION = float(os.getenv("MIN_HOURS_TO_RESOLUTION", "2"))
MAX_HOURS_TO_RESOLUTION = float(os.getenv("MAX_HOURS_TO_RESOLUTION", "72"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_TRADE_SIZE_USD = float(os.getenv("MAX_TRADE_SIZE_USD", "20.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))
TRAILING_STOP_TRIGGER = float(os.getenv("TRAILING_STOP_TRIGGER", "0.20"))

# ─── Simulation Parameters ────────────────────────────────────────────────────
SIM_ROUNDS = int(os.getenv("SIM_ROUNDS", "3"))
MAX_AGENTS_PER_SIM = 4       # weather analyst agents per city
CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.65"))
HIGH_SPREAD_THRESHOLD_F = float(os.getenv("HIGH_SPREAD_THRESHOLD_F", "8.0"))

# ─── Execution / Parallelism ─────────────────────────────────────────────────
# Max cities processed concurrently. Higher = faster cycle but more API load.
MAX_PARALLEL_CITIES = int(os.getenv("MAX_PARALLEL_CITIES", "4"))
# Max parallel order-book HTTP requests within a single city's market scan.
MAX_PARALLEL_ORDERBOOKS = int(os.getenv("MAX_PARALLEL_ORDERBOOKS", "8"))
# Hours-to-resolution threshold below which IOC orders are used instead of GTC.
IOC_URGENCY_HOURS = float(os.getenv("IOC_URGENCY_HOURS", "6.0"))
# Order submission retries (exponential backoff: 1s, 2s, 4s).
ORDER_RETRY_MAX = int(os.getenv("ORDER_RETRY_MAX", "3"))

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
PERSONA_WEIGHTING = os.getenv("PERSONA_WEIGHTING", "true").lower() != "false"
SAVE_DEBATE_TRANSCRIPTS = os.getenv("SAVE_DEBATE_TRANSCRIPTS", "true").lower() != "false"
SELF_PLAY_REFLECTION = os.getenv("SELF_PLAY_REFLECTION", "true").lower() != "false"
DEMO_SYNTHETIC_MARKETS = os.getenv("DEMO_SYNTHETIC_MARKETS", "true").lower() != "false"
MARKET_SCANNER_DEBUG = os.getenv("MARKET_SCANNER_DEBUG", "false").lower() == "true"
MAX_DEBATE_TRANSCRIPTS = int(os.getenv("MAX_DEBATE_TRANSCRIPTS", "200"))
# For real Polymarket markets, wait for official Gamma/CLOB resolution before
# settling paper/live positions. Synthetic demo markets still use weather
# archive fallback because they do not exist on Polymarket.
REQUIRE_OFFICIAL_POLYMARKET_RESOLUTION = (
    os.getenv("REQUIRE_OFFICIAL_POLYMARKET_RESOLUTION", "true").lower() != "false"
)
MARK_TO_MARKET_OPEN_POSITIONS = (
    os.getenv("MARK_TO_MARKET_OPEN_POSITIONS", "true").lower() != "false"
)
USE_RISK_REGIME_SIZING = (
    os.getenv("USE_RISK_REGIME_SIZING", "true").lower() != "false"
)
RISK_REGIME_FILE = os.getenv(
    "RISK_REGIME_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "btc_risk_regimes_1d.jsonl"),
)
DECISION_ENGINE_ENABLED = (
    os.getenv("DECISION_ENGINE_ENABLED", "true").lower() != "false"
)
DECISION_MIN_LIVE_FEATURE_QUALITY = float(os.getenv("DECISION_MIN_LIVE_FEATURE_QUALITY", "0.60"))
DECISION_LOW_DATA_RISK_MULTIPLIER = float(os.getenv("DECISION_LOW_DATA_RISK_MULTIPLIER", "0.50"))
DECISION_MAX_HOLDER_CONCENTRATION = float(os.getenv("DECISION_MAX_HOLDER_CONCENTRATION", "0.65"))
DECISION_CONTRA_FLOW_THRESHOLD = float(os.getenv("DECISION_CONTRA_FLOW_THRESHOLD", "0.35"))
DECISION_MAX_VOLATILITY_24H = float(os.getenv("DECISION_MAX_VOLATILITY_24H", "0.20"))
USE_WEATHER_NORMALS = (
    os.getenv("USE_WEATHER_NORMALS", "true").lower() != "false"
)
WEATHER_NORMALS_FILE = os.getenv(
    "WEATHER_NORMALS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "weather_city_month_normals.jsonl"),
)

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
UPDATE_INTERVAL_SECONDS = int(os.getenv("UPDATE_INTERVAL_SECONDS", "3600"))
