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

# Rich is already a dependency (tui uses it)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

_ENV_PATH = Path(__file__).parent / ".env"

# ─── Step helpers ─────────────────────────────────────────────────────────────

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


def _prompt_str(label: str, default: str = "", required: bool = False) -> str:
    hint = f" [dim](default: {default})[/]" if default else (
           " [dim](optional)[/]" if not required else "")
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
    val = _prompt_float(label, float(default), description, min_val, max_val)
    return int(val)


def _prompt_choice(label: str, choices: list[tuple[str, str]], default: str) -> str:
    """choices: list of (key, description). Returns selected key."""
    console.print(f"  [bold]{label}[/]")
    for key, desc in choices:
        marker = "[green]✓[/]" if key == default else " "
        console.print(f"    {marker} [bold]{key}[/] — {desc}")
    console.print(f"  [dim](default: {default})[/]")
    while True:
        try:
            raw = console.input("  → ").strip().lower()
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


# ─── .env reader ──────────────────────────────────────────────────────────────

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
    lines = [
        "# Albert Miro Weather — Configuration",
        "# Generated by setup wizard. Edit values here or re-run: python main.py --setup",
        "",
        "# ─── API Keys ────────────────────────────────────────────────────────",
    ]
    api_keys = ["ANTHROPIC_API_KEY", "POLYMARKET_API_KEY", "VISUAL_CROSSING_API_KEY"]
    for k in api_keys:
        if k in settings:
            lines.append(f"{k}={settings[k]}")

    lines += [
        "",
        "# ─── Trading Mode ────────────────────────────────────────────────────",
    ]
    if "DEFAULT_MODE" in settings:
        lines.append(f"DEFAULT_MODE={settings['DEFAULT_MODE']}")

    lines += [
        "",
        "# ─── Risk Parameters ────────────────────────────────────────────────",
    ]
    for k in ["MIN_EV", "KELLY_FRACTION", "MAX_TRADE_SIZE_USD", "STOP_LOSS_PCT", "TRAILING_STOP_TRIGGER"]:
        if k in settings:
            lines.append(f"{k}={settings[k]}")

    lines += [
        "",
        "# ─── Market Filters ─────────────────────────────────────────────────",
    ]
    for k in ["MIN_VOLUME", "MAX_SPREAD", "MIN_HOURS_TO_RESOLUTION", "MAX_HOURS_TO_RESOLUTION"]:
        if k in settings:
            lines.append(f"{k}={settings[k]}")

    lines += [
        "",
        "# ─── Simulation Settings ─────────────────────────────────────────────",
    ]
    for k in ["SIM_ROUNDS", "SCENARIO_SPECULATION", "HIGH_SPREAD_THRESHOLD_F", "UPDATE_INTERVAL_SECONDS", "CONSENSUS_THRESHOLD"]:
        if k in settings:
            lines.append(f"{k}={settings[k]}")

    with open(_ENV_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main wizard ──────────────────────────────────────────────────────────────

def run_wizard() -> None:
    existing = _read_existing_env()
    settings: dict[str, str] = {}
    STEPS = 6

    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]ALBERT MIRO WEATHER[/] — Setup Wizard\n\n"
            "This wizard configures Albert's API keys, trading mode, and parameters.\n"
            "Settings are saved to [bold].env[/] in the project directory.\n\n"
            "Press [bold]Enter[/] to accept defaults.  Press [bold]Ctrl+C[/] to cancel.",
        ),
        title="[bold]Welcome[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Step 1: API Keys ──────────────────────────────────────────────────────
    _header(1, STEPS, "API Keys")

    existing_ak = existing.get("ANTHROPIC_API_KEY", "")
    console.print("  [bold]ANTHROPIC_API_KEY[/] [red](required)[/]")
    if existing_ak:
        masked = existing_ak[:8] + "…" + existing_ak[-4:] if len(existing_ak) > 12 else "****"
        console.print(f"  [dim]Current value: {masked}[/]")
    ak = _prompt_secret("  Enter new key", required=not existing_ak, current=existing_ak)
    settings["ANTHROPIC_API_KEY"] = ak or existing_ak

    console.print()
    existing_pk = existing.get("POLYMARKET_API_KEY", "")
    console.print("  [bold]POLYMARKET_API_KEY[/] [dim](optional — needed for live trading)[/]")
    if existing_pk:
        masked = existing_pk[:8] + "…" if len(existing_pk) > 8 else "****"
        console.print(f"  [dim]Current value: {masked}[/]")
    pk = _prompt_secret("  Enter key", required=False)
    settings["POLYMARKET_API_KEY"] = pk or existing_pk

    console.print()
    existing_vc = existing.get("VISUAL_CROSSING_API_KEY", "")
    console.print("  [bold]VISUAL_CROSSING_API_KEY[/] [dim](optional — free Open-Meteo used as fallback)[/]")
    vc = _prompt_secret("  Enter key", required=False)
    settings["VISUAL_CROSSING_API_KEY"] = vc or existing_vc

    # ── Step 2: Trading Mode ──────────────────────────────────────────────────
    _header(2, STEPS, "Trading Mode")

    current_mode = existing.get("DEFAULT_MODE", "dry")
    mode = _prompt_choice(
        "Default mode when running Albert:",
        [
            ("dry",  "Dry-run — analyse only, never open positions (safe default)"),
            ("demo", "Demo — paper trading with virtual wallet, no real money"),
            ("live", "Live — execute real trades on Polymarket (requires API key)"),
        ],
        default=current_mode,
    )
    settings["DEFAULT_MODE"] = mode
    if mode == "live":
        console.print()
        console.print("  [bold yellow]⚠  Live mode will execute real trades.[/]")
        console.print("  [dim]You can always override at runtime: python main.py --live[/]")

    # ── Step 3: Risk Parameters ───────────────────────────────────────────────
    _header(3, STEPS, "Risk Parameters")

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
        description="Fractional Kelly cap — what fraction of the full Kelly stake to use.",
        min_val=0.01, max_val=1.0,
    ))
    console.print()
    settings["MAX_TRADE_SIZE_USD"] = str(_prompt_float(
        "MAX_TRADE_SIZE_USD",
        default=float(existing.get("MAX_TRADE_SIZE_USD", "20.0")),
        description="Maximum USD per single trade (hard cap regardless of Kelly).",
        min_val=1.0,
    ))
    console.print()
    settings["STOP_LOSS_PCT"] = str(_prompt_float(
        "STOP_LOSS_PCT",
        default=float(existing.get("STOP_LOSS_PCT", "0.20")),
        description="Stop-loss: close position when unrealized loss exceeds this fraction.",
        min_val=0.01, max_val=0.99,
    ))
    console.print()
    settings["TRAILING_STOP_TRIGGER"] = str(_prompt_float(
        "TRAILING_STOP_TRIGGER",
        default=float(existing.get("TRAILING_STOP_TRIGGER", "0.20")),
        description="Activate trailing stop after this fraction of profit (e.g. 0.20 = 20%).",
        min_val=0.01, max_val=0.99,
    ))

    # ── Step 4: Market Filters ────────────────────────────────────────────────
    _header(4, STEPS, "Market Filters")

    settings["MIN_VOLUME"] = str(_prompt_float(
        "MIN_VOLUME",
        default=float(existing.get("MIN_VOLUME", "500")),
        description="Minimum contract volume — skip illiquid markets below this.",
        min_val=0,
    ))
    console.print()
    settings["MAX_SPREAD"] = str(_prompt_float(
        "MAX_SPREAD",
        default=float(existing.get("MAX_SPREAD", "0.03")),
        description="Maximum bid-ask spread (fraction, e.g. 0.03 = 3%).",
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

    # ── Step 5: Simulation Settings ───────────────────────────────────────────
    _header(5, STEPS, "Simulation Settings")

    settings["SIM_ROUNDS"] = str(_prompt_int(
        "SIM_ROUNDS",
        default=int(existing.get("SIM_ROUNDS", "3")),
        description="Debate rounds per city in classic simulation mode (more = more API calls).",
        min_val=1, max_val=10,
    ))
    console.print()
    scenario_default = existing.get("SCENARIO_SPECULATION", "true").lower() != "false"
    scenario = _prompt_bool(
        "SCENARIO_SPECULATION",
        default=scenario_default,
        description="Use 3-phase Bayesian scenario tree (Morgan + River agents) instead of naive averaging.",
    )
    settings["SCENARIO_SPECULATION"] = "true" if scenario else "false"
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
        description="Seconds between daemon cycles (default: 3600 = 1 hour).",
        min_val=60,
    ))
    console.print()
    settings["CONSENSUS_THRESHOLD"] = str(_prompt_float(
        "CONSENSUS_THRESHOLD",
        default=float(existing.get("CONSENSUS_THRESHOLD", "0.65")),
        description="Probability threshold for 'high confidence' signal label.",
        min_val=0.5, max_val=0.99,
    ))

    # ── Step 6: Review & Save ─────────────────────────────────────────────────
    _header(6, STEPS, "Review & Save")

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Setting", style="bold", min_width=28)
    table.add_column("Value", style="green")

    sections = {
        "API Keys": ["ANTHROPIC_API_KEY", "POLYMARKET_API_KEY", "VISUAL_CROSSING_API_KEY"],
        "Trading Mode": ["DEFAULT_MODE"],
        "Risk Parameters": ["MIN_EV", "KELLY_FRACTION", "MAX_TRADE_SIZE_USD", "STOP_LOSS_PCT", "TRAILING_STOP_TRIGGER"],
        "Market Filters": ["MIN_VOLUME", "MAX_SPREAD", "MIN_HOURS_TO_RESOLUTION", "MAX_HOURS_TO_RESOLUTION"],
        "Simulation": ["SIM_ROUNDS", "SCENARIO_SPECULATION", "HIGH_SPREAD_THRESHOLD_F", "UPDATE_INTERVAL_SECONDS", "CONSENSUS_THRESHOLD"],
    }

    for section, keys in sections.items():
        table.add_row(f"[dim]── {section} ──[/]", "")
        for k in keys:
            v = settings.get(k, "")
            if not v:
                continue
            # Mask API keys
            if "KEY" in k and len(v) > 12:
                display = v[:8] + "…" + v[-4:]
            elif "KEY" in k and v:
                display = "****"
            else:
                display = v
            table.add_row(f"  {k}", display)

    console.print(table)
    console.print()

    if _ENV_PATH.exists():
        console.print(f"  [yellow]⚠  {_ENV_PATH} already exists and will be overwritten.[/]")

    save = _prompt_bool("Save these settings to .env?", default=True)
    if not save:
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
            "  [bold]python main.py --live --run[/]   live trading (real money)\n\n"
            "Re-run [bold]python main.py --setup[/] at any time to change settings."
        ),
        title="[bold green]Done[/]",
        border_style="green",
        padding=(1, 2),
    ))
