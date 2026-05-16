"""
Albert Miro Weather — Setup Wizard
Interactive terminal guide for first-time configuration.
Saves all settings to .env in the project root.
"""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:
    import re

    def _plain(markup: object) -> str:
        return re.sub(r"\[/?[^\]]+\]", "", str(markup))

    class Console:
        def print(self, *objects, **kwargs) -> None:
            print(*(_plain(obj) for obj in objects))

        def input(self, prompt: str = "") -> str:
            return input(_plain(prompt))

        def rule(self, title: str = "") -> None:
            clean = _plain(title)
            print(f"\n{'-' * 8} {clean} {'-' * 8}")

    class Panel:
        def __init__(self, renderable, *args, title: str = "", **kwargs) -> None:
            self.renderable = renderable
            self.title = title

        def __str__(self) -> str:
            title = _plain(self.title)
            body = _plain(self.renderable)
            return f"{title}\n{body}" if title else body

    class Table:
        def __init__(self, *args, **kwargs) -> None:
            self._rows: list[tuple[str, ...]] = []

        def add_column(self, *args, **kwargs) -> None:
            return None

        def add_row(self, *values) -> None:
            self._rows.append(tuple(_plain(v) for v in values))

        def __str__(self) -> str:
            return "\n".join("  ".join(row) for row in self._rows)

    class Text:
        @staticmethod
        def from_markup(markup: str):
            return _plain(markup)

console = Console()

_ENV_PATH = Path(__file__).parent / ".env"

# ─── Available LLM models ─────────────────────────────────────────────────────

_LLM_MODELS = [
    ("claude-sonnet-4-6",  "Sonnet 4.6  — fast, capable, cost-effective  (recommended)"),
    ("gpt-4.1",            "OpenAI GPT-4.1 — OpenAI-compatible APIs"),
    ("deepseek-chat",      "DeepSeek Chat — OpenAI-compatible APIs"),
    ("openrouter/auto",    "OpenRouter Auto — route to an available model"),
    ("claude-opus-4-7",    "Opus 4.7    — most capable, slower, higher cost"),
    ("claude-haiku-4-5",   "Haiku 4.5   — fastest, lowest cost, lighter analysis"),
    ("custom",             "Custom      — enter your own model ID"),
]

# ─── Prompt helpers ───────────────────────────────────────────────────────────

def _header(step: int, total: int, title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]Step {step}/{total} — {title}[/]")
    console.print()


def _prompt_secret(label: str, required: bool = True, current: str = "") -> str:
    """Hidden input (getpass). Returns empty string if optional and blank."""
    hint = " [dim](leave blank to keep existing)[/]" if current else (
           " [dim](optional, press Enter to skip)[/]" if not required else "")
    while True:
        console.print(f"  [bold]{label}[/]{hint}")
        try:
            val = getpass.getpass("  → ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Wizard cancelled.[/]")
            sys.exit(0)
        val = val.strip()
        if not val and required and not current:
            console.print("  [red]This field is required.[/]")
            continue
        return val


def _prompt_str(label: str, default: str = "", required: bool = False,
                description: str = "") -> str:
    hint = f" [dim](default: {default})[/]" if default else (
           " [dim](optional)[/]" if not required else "")
    if description:
        console.print(f"  [dim]{description}[/]")
    console.print(f"  [bold]{label}[/]{hint}")
    try:
        val = console.input("  → ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Wizard cancelled.[/]")
        sys.exit(0)
    return val or default


def _prompt_float(
    label: str,
    default: float,
    description: str = "",
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    bounds = ""
    if min_val is not None and max_val is not None:
        bounds = f" [{min_val}–{max_val}]"
    elif min_val is not None:
        bounds = f" [≥{min_val}]"
    elif max_val is not None:
        bounds = f" [≤{max_val}]"
    if description:
        console.print(f"  [dim]{description}[/]")
    console.print(f"  [bold]{label}[/] [dim](default: {default}){bounds}[/]")
    while True:
        try:
            raw = console.input("  → ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Wizard cancelled.[/]")
            sys.exit(0)
        if not raw:
            return default
        try:
            val = float(raw)
        except ValueError:
            console.print("  [red]Please enter a number.[/]")
            continue
        if min_val is not None and val < min_val:
            console.print(f"  [red]Value must be ≥ {min_val}.[/]")
            continue
        if max_val is not None and val > max_val:
            console.print(f"  [red]Value must be ≤ {max_val}.[/]")
            continue
        return val


def _prompt_int(
    label: str,
    default: int,
    description: str = "",
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    return int(_prompt_float(label, float(default), description, min_val, max_val))


def _prompt_choice(label: str, choices: list[tuple[str, str]], default: str) -> str:
    """choices: list of (key, description). Returns selected key."""
    console.print(f"  [bold]{label}[/]")
    for key, desc in choices:
        marker = "[green]✓[/]" if key == default else " "
        console.print(f"    {marker} [bold]{key}[/] — {desc}")
    console.print(f"  [dim](default: {default})[/]")
    while True:
        try:
            raw = console.input("  → ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Wizard cancelled.[/]")
            sys.exit(0)
        if not raw:
            return default
        valid = [k for k, _ in choices]
        if raw in valid:
            return raw
        console.print(f"  [red]Choose one of: {', '.join(valid)}[/]")


def _prompt_bool(label: str, default: bool, description: str = "") -> bool:
    if description:
        console.print(f"  [dim]{description}[/]")
    default_str = "Y/n" if default else "y/N"
    console.print(f"  [bold]{label}[/] [{default_str}]")
    while True:
        try:
            raw = console.input("  → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Wizard cancelled.[/]")
            sys.exit(0)
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        console.print("  [red]Please enter y or n.[/]")


def _mask(val: str, reveal_prefix: int = 8, reveal_suffix: int = 4) -> str:
    if not val:
        return "[dim](not set)[/]"
    if len(val) > reveal_prefix + reveal_suffix + 3:
        return val[:reveal_prefix] + "…" + val[-reveal_suffix:]
    return "****"


# ─── .env reader / writer ─────────────────────────────────────────────────────

def _read_existing_env() -> dict[str, str]:
    existing: dict[str, str] = {}
    if not _ENV_PATH.exists():
        return existing
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip().strip('"').strip("'")
    return existing


def _save_env(settings: dict[str, str]) -> None:
    sections = [
        ("# ─── LLM API ─────────────────────────────────────────────────────────────",
         ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL",
          "LLM_ENSEMBLE",
          "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "ANTHROPIC_BASE_URL"]),
        ("# ─── Polymarket Wallet ───────────────────────────────────────────────────",
         ["POLYMARKET_API_KEY", "POLYMARKET_PRIVATE_KEY", "POLYMARKET_PROXY_ADDRESS"]),
        ("# ─── Weather Data ────────────────────────────────────────────────────────",
         ["VISUAL_CROSSING_API_KEY"]),
        ("# ─── Trading Mode ────────────────────────────────────────────────────────",
         ["DEFAULT_MODE"]),
        ("# ─── Remote Control / Telegram ─────────────────────────────────────────",
         ["REMOTE_CONTROL_ENABLED", "REMOTE_CONTROL_PROVIDER", "TELEGRAM_BOT_TOKEN",
          "REMOTE_ALLOWED_CHAT_IDS", "REMOTE_ALLOWED_COMMANDS", "REMOTE_ALLOW_LIVE",
          "REMOTE_AUDIT_LOG", "REMOTE_POLL_INTERVAL_SECONDS",
          "REMOTE_DEMO_BALANCE", "REMOTE_DEMO_POSITIONS_FILE",
          "REMOTE_DEMO_TOKEN_BUDGET", "REMOTE_DEMO_SIM_ROUNDS"]),
        ("# ─── Risk Parameters ────────────────────────────────────────────────────",
         ["MIN_EV", "KELLY_FRACTION", "MAX_TRADE_SIZE_USD", "STOP_LOSS_PCT", "TRAILING_STOP_TRIGGER"]),
        ("# ─── Market Filters ─────────────────────────────────────────────────────",
         ["MIN_VOLUME", "MAX_SPREAD", "MIN_HOURS_TO_RESOLUTION", "MAX_HOURS_TO_RESOLUTION"]),
        ("# ─── Simulation Settings ─────────────────────────────────────────────────",
         ["SIM_ROUNDS", "SCENARIO_SPECULATION", "HIGH_SPREAD_THRESHOLD_F",
          "PERSONA_WEIGHTING", "SAVE_DEBATE_TRANSCRIPTS", "SELF_PLAY_REFLECTION",
          "UPDATE_INTERVAL_SECONDS", "CONSENSUS_THRESHOLD"]),
    ]

    lines = [
        "# Albert Miro Weather — Configuration",
        "# Generated by setup wizard. Edit values here or re-run: python main.py --setup",
    ]
    for header, keys in sections:
        lines += ["", header]
        for k in keys:
            v = settings.get(k, "")
            if v:
                lines.append(f"{k}={v}")

    with open(_ENV_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Wizard steps ─────────────────────────────────────────────────────────────

def _step_llm(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "LLM API")

    current_provider = existing.get("LLM_PROVIDER", "anthropic")
    provider = _prompt_choice(
        "LLM provider:",
        [
            ("anthropic", "Native Anthropic Messages API"),
            ("openai-compatible", "OpenAI-compatible /chat/completions API"),
            ("mock", "Offline deterministic responses for smoke tests only"),
        ],
        default=current_provider if current_provider in {"anthropic", "openai-compatible", "mock"} else "anthropic",
    )
    settings["LLM_PROVIDER"] = provider

    console.print()
    existing_key = existing.get("LLM_API_KEY") or existing.get("ANTHROPIC_API_KEY") or existing.get("OPENAI_API_KEY", "")
    if provider == "mock":
        settings["LLM_API_KEY"] = ""
    else:
        console.print("  [bold]LLM_API_KEY[/] [red](required)[/]")
        if existing_key:
            console.print(f"  [dim]Current: {_mask(existing_key)}[/]")
        key = _prompt_secret("  Enter key", required=not existing_key, current=existing_key)
        settings["LLM_API_KEY"] = key or existing_key

    console.print()
    current_model = existing.get("LLM_MODEL") or existing.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    choice = _prompt_choice(
        "Model to use for all agents:",
        _LLM_MODELS,
        default=current_model if current_model in dict(_LLM_MODELS) else "custom",
    )
    if choice == "custom":
        model = _prompt_str(
            "  Model ID",
            default=current_model,
            required=True,
            description="Enter the exact model ID required by your provider.",
        )
    else:
        model = choice
    settings["LLM_MODEL"] = model

    console.print()
    existing_url = existing.get("LLM_BASE_URL") or existing.get("ANTHROPIC_BASE_URL", "")
    if provider == "mock":
        settings["LLM_BASE_URL"] = ""
    elif provider == "openai-compatible":
        url = _prompt_str(
            "  LLM_BASE_URL",
            default=existing_url or "https://api.openai.com/v1",
            required=True,
            description="Base URL ending before /chat/completions, e.g. https://api.openai.com/v1",
        )
        settings["LLM_BASE_URL"] = url.rstrip("/")
        console.print(f"  [dim]Requests will go to: {url}[/]")
    else:
        use_custom_url = _prompt_bool(
            "Use a custom Anthropic base URL?",
            default=bool(existing_url),
            description="Enable this for Anthropic-compatible proxies or private deployments.",
        )
        settings["LLM_BASE_URL"] = (
            _prompt_str("  LLM_BASE_URL", default=existing_url, required=True)
            if use_custom_url else ""
        )

    # Backward-compatible aliases for older scripts.
    if provider == "anthropic":
        settings["ANTHROPIC_API_KEY"] = settings["LLM_API_KEY"]
        settings["CLAUDE_MODEL"] = settings["LLM_MODEL"]
        settings["ANTHROPIC_BASE_URL"] = settings["LLM_BASE_URL"]

    console.print()
    ensemble = _prompt_str(
        "  LLM_ENSEMBLE",
        default=existing.get("LLM_ENSEMBLE", ""),
        required=False,
        description="Optional comma-separated provider:model:base_url list for multi-provider persona ensemble.",
    )
    settings["LLM_ENSEMBLE"] = ensemble


def _step_wallet(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Polymarket Wallet")

    console.print("  [dim]The API key is for read access and market scanning.[/]")
    console.print("  [dim]The private key is required to sign and submit live orders.[/]")
    console.print()

    # CLOB API key
    existing_pk = existing.get("POLYMARKET_API_KEY", "")
    console.print("  [bold]POLYMARKET_API_KEY[/] [dim](optional — market data + live trading)[/]")
    if existing_pk:
        console.print(f"  [dim]Current: {_mask(existing_pk)}[/]")
    pk = _prompt_secret("  Enter key", required=False)
    settings["POLYMARKET_API_KEY"] = pk or existing_pk

    # Private key (wallet signing key)
    console.print()
    existing_priv = existing.get("POLYMARKET_PRIVATE_KEY", "")
    console.print("  [bold]POLYMARKET_PRIVATE_KEY[/] [dim](required for live order signing)[/]")
    console.print("  [dim]Ethereum private key (hex, with or without 0x prefix).[/]")
    console.print("  [bold yellow]  ⚠  Never share this key. It controls your wallet.[/]")
    if existing_priv:
        console.print(f"  [dim]Current: {_mask(existing_priv, 6, 4)}[/]")
    priv = _prompt_secret("  Enter private key", required=False)
    settings["POLYMARKET_PRIVATE_KEY"] = priv or existing_priv

    # Proxy wallet address (optional)
    console.print()
    existing_proxy = existing.get("POLYMARKET_PROXY_ADDRESS", "")
    use_proxy = _prompt_bool(
        "Use a Polymarket proxy wallet address?",
        default=bool(existing_proxy),
        description="Only needed if your account uses a Polymarket proxy contract (most users: No).",
    )
    if use_proxy:
        proxy = _prompt_str(
            "  POLYMARKET_PROXY_ADDRESS",
            default=existing_proxy,
            required=True,
            description="Ethereum address of the proxy contract (0x…).",
        )
        settings["POLYMARKET_PROXY_ADDRESS"] = proxy
    else:
        settings["POLYMARKET_PROXY_ADDRESS"] = existing_proxy

    # Visual Crossing (weather data, minor — fold in here)
    console.print()
    existing_vc = existing.get("VISUAL_CROSSING_API_KEY", "")
    console.print("  [bold]VISUAL_CROSSING_API_KEY[/] [dim](optional — free Open-Meteo used as fallback)[/]")
    if existing_vc:
        console.print(f"  [dim]Current: {_mask(existing_vc)}[/]")
    vc = _prompt_secret("  Enter key", required=False)
    settings["VISUAL_CROSSING_API_KEY"] = vc or existing_vc


def _step_trading_mode(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Trading Mode")

    current_mode = existing.get("DEFAULT_MODE", "dry")
    mode = _prompt_choice(
        "Default mode when running Albert:",
        [
            ("dry",  "Dry-run — analyse only, never open positions (safe default)"),
            ("demo", "Demo — paper trading with virtual wallet, no real money"),
            ("live", "Live — execute real trades on Polymarket (requires wallet key)"),
        ],
        default=current_mode,
    )
    settings["DEFAULT_MODE"] = mode
    if mode == "live":
        console.print()
        console.print("  [bold yellow]⚠  Live mode will execute real trades using your private key.[/]")
        console.print("  [dim]Override at runtime with: python main.py --live  or  --demo[/]")


def _step_remote_control(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Remote Control / Telegram")

    enabled = _prompt_bool(
        "Enable Telegram remote control?",
        default=existing.get("REMOTE_CONTROL_ENABLED", "false").lower() == "true",
        description="Default is off. Remote commands are allowlisted and live trading is blocked by default.",
    )
    settings["REMOTE_CONTROL_ENABLED"] = "true" if enabled else "false"
    settings["REMOTE_CONTROL_PROVIDER"] = "telegram"

    existing_token = existing.get("TELEGRAM_BOT_TOKEN", "")
    console.print()
    console.print("  [bold]TELEGRAM_BOT_TOKEN[/] [dim](from @BotFather)[/]")
    if existing_token:
        console.print(f"  [dim]Current: {_mask(existing_token)}[/]")
    token = _prompt_secret("  Enter bot token", required=enabled and not existing_token, current=existing_token)
    settings["TELEGRAM_BOT_TOKEN"] = token or existing_token

    console.print()
    settings["REMOTE_ALLOWED_CHAT_IDS"] = _prompt_str(
        "REMOTE_ALLOWED_CHAT_IDS",
        default=existing.get("REMOTE_ALLOWED_CHAT_IDS", ""),
        required=False,
        description="Comma-separated Telegram chat IDs allowed to control Albert. Use /whoami to discover yours.",
    )
    console.print()
    settings["REMOTE_ALLOWED_COMMANDS"] = _prompt_str(
        "REMOTE_ALLOWED_COMMANDS",
        default=existing.get(
            "REMOTE_ALLOWED_COMMANDS",
            "status,positions,signals,learning,pause,resume,dry_run_once,demo_once",
        ),
        required=False,
        description="Comma-separated command allowlist. Keep live commands out for safety.",
    )
    console.print()
    allow_live = _prompt_bool(
        "REMOTE_ALLOW_LIVE",
        default=existing.get("REMOTE_ALLOW_LIVE", "false").lower() == "true",
        description="Dangerous: allows future live remote commands. Recommended: No.",
    )
    settings["REMOTE_ALLOW_LIVE"] = "true" if allow_live else "false"
    if allow_live:
        console.print("  [bold yellow]⚠  Review wallet exposure before enabling any live remote command.[/]")

    console.print()
    settings["REMOTE_AUDIT_LOG"] = _prompt_str(
        "REMOTE_AUDIT_LOG",
        default=existing.get("REMOTE_AUDIT_LOG", "remote_control.log"),
        required=False,
        description="JSONL log for every remote command attempt.",
    )
    console.print()
    settings["REMOTE_POLL_INTERVAL_SECONDS"] = str(_prompt_float(
        "REMOTE_POLL_INTERVAL_SECONDS",
        default=float(existing.get("REMOTE_POLL_INTERVAL_SECONDS", "2.0")),
        description="Seconds between Telegram polling retries after empty/error responses.",
        min_val=0.5,
    ))
    console.print()
    settings["REMOTE_DEMO_BALANCE"] = str(_prompt_float(
        "REMOTE_DEMO_BALANCE",
        default=float(existing.get("REMOTE_DEMO_BALANCE", "1000.0")),
        description="Virtual wallet balance used by /demo_once.",
        min_val=1.0,
    ))
    console.print()
    settings["REMOTE_DEMO_POSITIONS_FILE"] = _prompt_str(
        "REMOTE_DEMO_POSITIONS_FILE",
        default=existing.get("REMOTE_DEMO_POSITIONS_FILE", ".demo_runs/telegram_demo_positions.json"),
        required=False,
        description="Separate demo position state for Telegram-triggered demo runs.",
    )
    console.print()
    settings["REMOTE_DEMO_TOKEN_BUDGET"] = str(_prompt_int(
        "REMOTE_DEMO_TOKEN_BUDGET",
        default=int(existing.get("REMOTE_DEMO_TOKEN_BUDGET", "200000")),
        description="Input token warning budget for /demo_once.",
        min_val=1000,
    ))
    console.print()
    settings["REMOTE_DEMO_SIM_ROUNDS"] = str(_prompt_int(
        "REMOTE_DEMO_SIM_ROUNDS",
        default=int(existing.get("REMOTE_DEMO_SIM_ROUNDS", "1")),
        description="Debate rounds per city for /demo_once.",
        min_val=1,
        max_val=10,
    ))


def _step_risk(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Risk Parameters")

    settings["MIN_EV"] = str(_prompt_float(
        "MIN_EV",
        default=float(existing.get("MIN_EV", "0.10")),
        description="Minimum expected value to enter a trade (higher = more selective).",
        min_val=0.01, max_val=1.0,
    ))
    console.print()
    settings["KELLY_FRACTION"] = str(_prompt_float(
        "KELLY_FRACTION",
        default=float(existing.get("KELLY_FRACTION", "0.25")),
        description="Fractional Kelly cap — fraction of the full Kelly stake to actually use.",
        min_val=0.01, max_val=1.0,
    ))
    console.print()
    settings["MAX_TRADE_SIZE_USD"] = str(_prompt_float(
        "MAX_TRADE_SIZE_USD",
        default=float(existing.get("MAX_TRADE_SIZE_USD", "20.0")),
        description="Hard cap on USD per single trade, regardless of Kelly.",
        min_val=1.0,
    ))
    console.print()
    settings["STOP_LOSS_PCT"] = str(_prompt_float(
        "STOP_LOSS_PCT",
        default=float(existing.get("STOP_LOSS_PCT", "0.20")),
        description="Close position when unrealized loss exceeds this fraction (0.20 = 20%).",
        min_val=0.01, max_val=0.99,
    ))
    console.print()
    settings["TRAILING_STOP_TRIGGER"] = str(_prompt_float(
        "TRAILING_STOP_TRIGGER",
        default=float(existing.get("TRAILING_STOP_TRIGGER", "0.20")),
        description="Activate trailing stop after this fraction of profit (0.20 = 20%).",
        min_val=0.01, max_val=0.99,
    ))


def _step_filters(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Market Filters")

    settings["MIN_VOLUME"] = str(_prompt_float(
        "MIN_VOLUME",
        default=float(existing.get("MIN_VOLUME", "500")),
        description="Skip illiquid markets below this contract volume.",
        min_val=0,
    ))
    console.print()
    settings["MAX_SPREAD"] = str(_prompt_float(
        "MAX_SPREAD",
        default=float(existing.get("MAX_SPREAD", "0.03")),
        description="Skip wide-spread markets above this bid-ask fraction (0.03 = 3%).",
        min_val=0.001, max_val=0.5,
    ))
    console.print()
    settings["MIN_HOURS_TO_RESOLUTION"] = str(_prompt_float(
        "MIN_HOURS_TO_RESOLUTION",
        default=float(existing.get("MIN_HOURS_TO_RESOLUTION", "2")),
        description="Skip markets resolving in fewer hours than this.",
        min_val=0,
    ))
    console.print()
    settings["MAX_HOURS_TO_RESOLUTION"] = str(_prompt_float(
        "MAX_HOURS_TO_RESOLUTION",
        default=float(existing.get("MAX_HOURS_TO_RESOLUTION", "72")),
        description="Skip markets resolving further out than this many hours.",
        min_val=1,
    ))


def _step_simulation(existing: dict, settings: dict, step: int, total: int) -> None:
    _header(step, total, "Simulation Settings")

    settings["SIM_ROUNDS"] = str(_prompt_int(
        "SIM_ROUNDS",
        default=int(existing.get("SIM_ROUNDS", "3")),
        description="Debate rounds per city in classic mode (each round = API calls per analyst).",
        min_val=1, max_val=10,
    ))
    console.print()
    scenario_default = existing.get("SCENARIO_SPECULATION", "true").lower() != "false"
    scenario = _prompt_bool(
        "SCENARIO_SPECULATION",
        default=scenario_default,
        description="3-phase Bayesian scenario tree (Morgan + River) — more accurate, ~6 API calls vs 12.",
    )
    settings["SCENARIO_SPECULATION"] = "true" if scenario else "false"
    console.print()
    settings["PERSONA_WEIGHTING"] = "true" if _prompt_bool(
        "PERSONA_WEIGHTING",
        default=existing.get("PERSONA_WEIGHTING", "true").lower() != "false",
        description="Weight analyst votes by historical Brier score once outcomes resolve.",
    ) else "false"
    console.print()
    settings["SAVE_DEBATE_TRANSCRIPTS"] = "true" if _prompt_bool(
        "SAVE_DEBATE_TRANSCRIPTS",
        default=existing.get("SAVE_DEBATE_TRANSCRIPTS", "true").lower() != "false",
        description="Persist full multi-agent debate transcripts for audit and self-play.",
    ) else "false"
    console.print()
    settings["SELF_PLAY_REFLECTION"] = "true" if _prompt_bool(
        "SELF_PLAY_REFLECTION",
        default=existing.get("SELF_PLAY_REFLECTION", "true").lower() != "false",
        description="Run MiroFish-style reflection over recent debate transcripts.",
    ) else "false"
    console.print()
    settings["HIGH_SPREAD_THRESHOLD_F"] = str(_prompt_float(
        "HIGH_SPREAD_THRESHOLD_F",
        default=float(existing.get("HIGH_SPREAD_THRESHOLD_F", "8.0")),
        description="°F model disagreement above which confidence is forced to 'low'.",
        min_val=0.0,
    ))
    console.print()
    settings["UPDATE_INTERVAL_SECONDS"] = str(_prompt_int(
        "UPDATE_INTERVAL_SECONDS",
        default=int(existing.get("UPDATE_INTERVAL_SECONDS", "3600")),
        description="Seconds between daemon cycles (3600 = 1 hour).",
        min_val=60,
    ))
    console.print()
    settings["CONSENSUS_THRESHOLD"] = str(_prompt_float(
        "CONSENSUS_THRESHOLD",
        default=float(existing.get("CONSENSUS_THRESHOLD", "0.65")),
        description="Probability threshold for 'high confidence' signal label.",
        min_val=0.5, max_val=0.99,
    ))


def _step_review(settings: dict, step: int, total: int) -> bool:
    _header(step, total, "Review & Save")

    # Sensitive keys — shown masked; everything else shown plainly
    _SECRET_KEYS = {"LLM_API_KEY", "ANTHROPIC_API_KEY", "POLYMARKET_API_KEY",
                    "POLYMARKET_PRIVATE_KEY", "VISUAL_CROSSING_API_KEY",
                    "TELEGRAM_BOT_TOKEN"}

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Setting", style="bold", min_width=30)
    table.add_column("Value", style="green")

    display_sections = {
        "LLM API":           ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_ENSEMBLE"],
        "Polymarket Wallet":  ["POLYMARKET_API_KEY", "POLYMARKET_PRIVATE_KEY", "POLYMARKET_PROXY_ADDRESS"],
        "Weather Data":       ["VISUAL_CROSSING_API_KEY"],
        "Trading Mode":       ["DEFAULT_MODE"],
        "Remote Control":     ["REMOTE_CONTROL_ENABLED", "REMOTE_CONTROL_PROVIDER",
                               "TELEGRAM_BOT_TOKEN", "REMOTE_ALLOWED_CHAT_IDS",
                               "REMOTE_ALLOWED_COMMANDS", "REMOTE_ALLOW_LIVE",
                               "REMOTE_AUDIT_LOG", "REMOTE_POLL_INTERVAL_SECONDS",
                               "REMOTE_DEMO_BALANCE", "REMOTE_DEMO_POSITIONS_FILE",
                               "REMOTE_DEMO_TOKEN_BUDGET", "REMOTE_DEMO_SIM_ROUNDS"],
        "Risk Parameters":    ["MIN_EV", "KELLY_FRACTION", "MAX_TRADE_SIZE_USD", "STOP_LOSS_PCT", "TRAILING_STOP_TRIGGER"],
        "Market Filters":     ["MIN_VOLUME", "MAX_SPREAD", "MIN_HOURS_TO_RESOLUTION", "MAX_HOURS_TO_RESOLUTION"],
        "Simulation":         ["SIM_ROUNDS", "SCENARIO_SPECULATION", "HIGH_SPREAD_THRESHOLD_F",
                               "PERSONA_WEIGHTING", "SAVE_DEBATE_TRANSCRIPTS",
                               "SELF_PLAY_REFLECTION", "UPDATE_INTERVAL_SECONDS",
                               "CONSENSUS_THRESHOLD"],
    }

    for section, keys in display_sections.items():
        table.add_row(f"[dim]── {section} ──[/]", "")
        for k in keys:
            v = settings.get(k, "")
            if not v:
                table.add_row(f"  {k}", "[dim](not set)[/]")
                continue
            display = _mask(v) if k in _SECRET_KEYS else v
            table.add_row(f"  {k}", display)

    console.print(table)
    console.print()

    if _ENV_PATH.exists():
        console.print(f"  [yellow]⚠  {_ENV_PATH} already exists and will be overwritten.[/]")

    return _prompt_bool("Save these settings to .env?", default=True)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_wizard() -> None:
    existing = _read_existing_env()
    settings: dict[str, str] = {}
    STEPS = 8

    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]ALBERT MIRO WEATHER[/] — Setup Wizard\n\n"
            "Configures API keys, wallet, LLM model, trading mode, and all parameters.\n"
            "Settings are saved to [bold].env[/] in the project directory.\n\n"
            "Press [bold]Enter[/] to accept defaults.  Press [bold]Ctrl+C[/] to cancel.",
        ),
        title="[bold]Welcome[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    _step_llm(existing, settings, 1, STEPS)
    _step_wallet(existing, settings, 2, STEPS)
    _step_trading_mode(existing, settings, 3, STEPS)
    _step_remote_control(existing, settings, 4, STEPS)
    _step_risk(existing, settings, 5, STEPS)
    _step_filters(existing, settings, 6, STEPS)
    _step_simulation(existing, settings, 7, STEPS)

    if not _step_review(settings, 8, STEPS):
        console.print("\n  [yellow]Settings not saved. Run the wizard again to configure.[/]")
        return

    _save_env(settings)
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[green bold]✓ Configuration saved to {_ENV_PATH}[/]\n\n"
            "To start Albert:\n"
            "  [bold]python main.py --run[/]         one cycle, dry-run\n"
            "  [bold]python main.py --daemon[/]       continuous, dry-run\n"
            "  [bold]python main.py --tui[/]          real-time dashboard\n"
            "  [bold]python main.py --demo[/]         paper-trading demo\n"
            "  [bold]python main.py --telegram-control[/] remote Telegram control\n"
            "  [bold]python main.py --live --run[/]   live trading (real money)\n\n"
            "Re-run [bold]python main.py --setup[/] at any time to change settings."
        ),
        title="[bold green]Done[/]",
        border_style="green",
        padding=(1, 2),
    ))
