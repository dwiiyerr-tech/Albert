"""
Albert command launcher.

Run `albert` from the shell to open an operational menu for setup, control,
demo runs, monitoring, and data utilities.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

SECRET_KEYS = {
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_PRIVATE_KEY",
    "TELEGRAM_BOT_TOKEN",
    "VISUAL_CROSSING_API_KEY",
}

SUPPORTED_LLM_PROVIDERS = {
    "anthropic",
    "openai",
    "openai-compatible",
    "compatible",
    "mock",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    os.chdir(ROOT)

    if argv:
        return run_subcommand(argv)

    return menu()


def run_subcommand(argv: list[str]) -> int:
    command = argv[0].lower().replace("-", "_")
    rest = argv[1:]

    aliases = {
        "help": print_help,
        "menu": lambda: menu(),
        "setup": lambda: run_python(["main.py", "--setup"]),
        "config": show_config,
        "status": show_config,
        "run": lambda: run_python(["main.py", "--run", *rest]),
        "dry": lambda: run_python(["main.py", "--demo", "--demo-cycles", "1", *rest]),
        "daemon": lambda: run_python(["main.py", "--daemon", *rest]),
        "tui": lambda: run_python(["main.py", "--tui", *rest]),
        "dashboard": lambda: run_python(["main.py", "--tui", *rest]),
        "control": lambda: run_python(["main.py", "--telegram-control", *rest]),
        "telegram": lambda: run_python(["main.py", "--telegram-control", *rest]),
        "demo": lambda: run_python(["main.py", "--demo", *rest]),
        "demo_once": lambda: run_python(["main.py", "--demo", "--demo-cycles", "1", *rest]),
        "positions": lambda: run_python(["main.py", "--positions", *rest]),
        "learning": lambda: run_python(["main.py", "--learning-status", *rest]),
        "scorecard": lambda: run_python(["scorecard.py", *rest]),
        "health": lambda: run_python(["healthcheck.py", *rest]),
        "healthcheck": lambda: run_python(["healthcheck.py", *rest]),
        "reflect": lambda: run_python(["main.py", "--reflect", *rest]),
        "ingest": lambda: run_python(["ingest_data.py", *rest]),
        "refresh_data": lambda: run_python(["ingest_data.py", "--source", "polymarket-pro-refresh", *rest]),
        "refresh-data": lambda: run_python(["ingest_data.py", "--source", "polymarket-pro-refresh", *rest]),
        "build_features": lambda: run_python(["build_features.py", *rest]),
        "git": lambda: run(["git", *rest]),
    }

    handler = aliases.get(command)
    if handler is None:
        print(f"Unknown albert command: {argv[0]}")
        print_help()
        return 2
    return int(handler() or 0)


def menu() -> int:
    while True:
        print()
        print("ALBERT CONTROL")
        print("=" * 48)
        print("1. Setup wizard")
        print("2. Show current config")
        print("3. Run one demo/dry paper cycle")
        print("4. Run one isolated demo cycle")
        print("5. Run demo mode")
        print("6. Open TUI dashboard")
        print("7. Start Telegram remote control")
        print("8. Start daemon")
        print("9. Show positions")
        print("10. Show learning status")
        print("11. Show decision scorecard")
        print("12. Refresh Polymarket pro data")
        print("13. List data sources")
        print("14. Build processed features")
        print("15. Run healthcheck")
        print("16. Git status")
        print("0. Exit")
        print()

        choice = input("Choose: ").strip().lower()
        print()

        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            run_python(["main.py", "--setup"])
        elif choice == "2":
            show_config()
        elif choice == "3":
            run_python(["main.py", "--demo", "--demo-cycles", "1"])
        elif choice == "4":
            run_python(["main.py", "--demo", "--demo-cycles", "1"])
        elif choice == "5":
            run_python(["main.py", "--demo"])
        elif choice == "6":
            run_python(["main.py", "--tui"])
        elif choice == "7":
            run_python(["main.py", "--telegram-control"])
        elif choice == "8":
            run_python(["main.py", "--daemon"])
        elif choice == "9":
            run_python(["main.py", "--positions"])
        elif choice == "10":
            run_python(["main.py", "--learning-status"])
        elif choice == "11":
            run_python(["scorecard.py"])
        elif choice == "12":
            run_python(["ingest_data.py", "--source", "polymarket-pro-refresh", "--append"])
        elif choice == "13":
            run_python(["ingest_data.py", "--list-sources"])
        elif choice == "14":
            run_python(["build_features.py", "--source", "all"])
        elif choice == "15":
            run_python(["healthcheck.py"])
        elif choice == "16":
            run(["git", "status", "--short", "--branch"])
        else:
            print("Unknown choice.")

        pause()


def run_python(args: list[str]) -> int:
    return run([sys.executable, *args])


def run(args: list[str]) -> int:
    print("$ " + " ".join(shlex.quote(arg) for arg in args))
    try:
        completed = subprocess.run(args, cwd=ROOT)
    except FileNotFoundError as exc:
        print(f"Command not found: {exc.filename}")
        return 127
    return int(completed.returncode)


def show_config() -> int:
    values = read_env()
    print("Project:", ROOT)
    print(".env:", ENV_PATH if ENV_PATH.exists() else "not found")
    print()

    keys = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "DEFAULT_MODE",
        "LIVE_TRADING_ENABLED",
        "DEMO_ONLY_UNTIL",
        "REMOTE_CONTROL_ENABLED",
        "TELEGRAM_BOT_TOKEN",
        "REMOTE_ALLOWED_CHAT_IDS",
        "REMOTE_ALLOW_LIVE",
        "POLYMARKET_API_KEY",
        "POLYMARKET_PRIVATE_KEY",
    ]
    for key in keys:
        value = values.get(key, "")
        if key in SECRET_KEYS:
            value = mask(value)
        elif key == "LLM_PROVIDER" and value and value.lower() not in SUPPORTED_LLM_PROVIDERS:
            value = "(invalid unsupported provider, hidden)"
        elif not value:
            value = "(not set)"
        print(f"{key}={value}")
    return 0


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mask(value: str) -> str:
    if not value:
        return "(not set)"
    return "(set, hidden)"


def pause() -> None:
    try:
        input("\nPress Enter to continue...")
    except (EOFError, KeyboardInterrupt):
        print()


def print_help() -> int:
    print("Usage:")
    print("  albert                 open interactive menu")
    print("  albert setup           run setup wizard")
    print("  albert config          show masked config")
    print("  albert run             run one analysis cycle")
    print("  albert dry             run one demo/paper cycle")
    print("  albert demo            run demo mode")
    print("  albert demo-once       run one demo cycle")
    print("  albert tui             open dashboard")
    print("  albert control         start Telegram remote control")
    print("  albert daemon          start daemon")
    print("  albert positions       show positions")
    print("  albert learning        show learning status")
    print("  albert scorecard       score resolved decisions and trades")
    print("  albert health          check config, LLM, and external APIs")
    print("  albert refresh-data    collect Polymarket pro data for top active markets")
    print("  albert ingest ARGS     pass args to ingest_data.py")
    print("  albert build-features  build processed features")
    print("  albert git ARGS        pass args to git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
