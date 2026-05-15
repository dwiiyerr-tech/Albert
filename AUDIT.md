# Albert / MiroWeather Audit

Audit date: 2026-05-15
Remediation status: patched locally on branch `audit-claude`

## Repository State

- Source: https://github.com/dwiiyerr-tech/Albert
- Local path: `/data/data/com.termux/files/home/Albert`
- Default branch: `main`
- Active audit branch: `audit-claude`, tracking `origin/claude/create-combined-agent-NqBfw`

`main` currently contains only `README.md`. The functional application is on
`origin/claude/create-combined-agent-NqBfw`.

## Project Summary

The active branch is a Python application named MiroWeather. It combines:

- weather forecast ingestion from Open-Meteo and Aviation Weather
- Polymarket weather-market scanning
- Anthropic/Claude multi-agent weather simulation
- expected value and Kelly sizing
- paper/demo/live trading paths
- persistent learning memory, calibration, and market-pattern learning
- rich terminal dashboard and setup wizard

Main entry point: `main.py`

Core modules:

- `config.py`: environment and strategy configuration
- `weather_data.py`: forecast and historical temperature data
- `simulation/agents.py`: multi-agent / scenario simulation
- `simulation/report_generator.py`: Claude-generated reports
- `trading/market_scanner.py`: Polymarket market discovery
- `trading/ev_calculator.py`: EV and position sizing
- `trading/order_executor.py`: live Polymarket CLOB orders
- `trading/position_manager.py`: paper/live position persistence
- `learning/*`: memory, calibration, reflection, market learner
- `setup_wizard.py`: interactive `.env` setup
- `tui/*`: terminal dashboard

## Verification Performed

- Cloned repository successfully.
- Listed all branches.
- Checked out active development branch locally as `audit-claude`.
- Reviewed main runtime, config, weather, trading, simulation, learning, setup, and reporting modules.
- Ran syntax validation with `python -m compileall .`; it passed.

## High Priority Findings

### 1. Setup mode is blocked by missing API key

Status: fixed.

`config.py` requires `ANTHROPIC_API_KEY` at import time. `main.py` imports config before parsing `--setup`, so a new user cannot run the setup wizard:

```text
ERROR: required environment variable 'ANTHROPIC_API_KEY' is not set. Run 'python main.py --setup' to configure Albert.
```

Impact: first-run onboarding is broken. Demo and other non-live modes are also blocked unless a key already exists.

Recommended fix:

- Do not call `_require_env("ANTHROPIC_API_KEY")` during `config.py` import.
- Validate the key only when constructing Claude clients or starting a mode that actually needs LLM calls.
- Let `python main.py --setup` import only `setup_wizard.py` before config validation.

Implemented:

- `config.py` now loads `ANTHROPIC_API_KEY` lazily.
- Claude-backed components validate the key only when instantiated.
- `main.py --setup` now starts without an existing `.env`.
- `setup_wizard.py` has a plain terminal fallback if `rich` is not installed.

### 2. Live mode can record a live position without a submitted order

Status: fixed.

In `main.py`, when `--live` is used and `self.executor` exists but is not configured, the code skips order placement and still calls `self.positions.open_position(...)`.

Impact: the system can report a "LIVE TRADE" and persist an open position even though no Polymarket order was submitted.

Recommended fix:

- In live mode, require `self.executor.is_configured()` before opening any position.
- If the executor is not configured, skip the signal and log a hard warning.
- Consider failing startup for `--live` without a configured private key/client.

Implemented:

- Live startup now fails if the Polymarket executor is not configured.
- Live signal execution now skips position creation unless `place_order()` returns an order ID.

### 3. Calibration status rendering contains invalid f-string format logic

Status: fixed.

`learning/calibration.py` contains conditional expressions inside format specs:

```python
f"Brier={brier:.4f if brier else '  N/A'}  "
f"ECE={ece:.4f if ece else '  N/A'}"
```

Impact: `calibration_report()` can raise a formatting exception once city breakdown rows are reached.

Recommended fix:

- Precompute display strings:

```python
brier_s = f"{brier:.4f}" if brier is not None else "N/A"
ece_s = f"{ece:.4f}" if ece is not None else "N/A"
```

Implemented in `learning/calibration.py`.

### 4. `EVCalculator.evaluate()` references an undefined logger

Status: fixed.

`trading/ev_calculator.py` calls `logger.warning(...)`, but the module does not define `logger`.

Impact: invalid simulation probabilities raise `NameError` instead of being skipped cleanly.

Recommended fix:

- Add `import logging` at module scope.
- Add `logger = logging.getLogger(__name__)`.

Implemented in `trading/ev_calculator.py`.

## Medium Priority Findings

### 5. Compiled Python artifacts are tracked in the branch

Status: fixed locally.

The branch contains `__pycache__` and `.pyc` files under `demo/` and `tui/`.

Impact: noisy diffs, platform-specific artifacts, and unnecessary repository churn.

Recommended fix:

- Remove tracked `.pyc` files.
- Ensure `.gitignore` contains `__pycache__/` and `*.pyc`.

Implemented:

- Removed tracked `.pyc` files from Git.
- Existing `.gitignore` already excludes Python bytecode artifacts.

### 6. `DEFAULT_MODE` is saved but not used by CLI startup

Status: fixed.

The setup wizard asks for `DEFAULT_MODE`, but `main.py` does not appear to use it when no mode flag is passed.

Impact: user configuration does not affect default runtime behavior.

Recommended fix:

- Read `DEFAULT_MODE` from config.
- Apply it when neither `--run`, `--daemon`, `--demo`, nor `--live` is provided, or remove the setting from setup.

Implemented:

- `DEFAULT_MODE=demo` now applies to `--run` / `--daemon` when no explicit mode flag is provided.
- `DEFAULT_MODE=live` now applies to `--run`, `--daemon`, and `--tui` when no explicit mode flag is provided.

### 7. City-level reflection uses unresolved records to infer freshly resolved cities

Status: fixed.

`main.py` builds `cities_resolved` from `self.memory.unresolved_records()` immediately after resolving old predictions.

Impact: city-level reflection may target the wrong city set or skip the newly resolved cities.

Recommended fix:

- Have `_resolve_pending_predictions()` return the set of newly resolved cities in addition to count.
- Use that set in `_run_learning_cycle()`.

Implemented in `main.py`.

### 8. Non-LLM commands import Claude dependencies

Status: fixed.

`main.py --positions` and setup paths previously imported `simulation.agents`
and `learning.reflection` at startup, which required `anthropic` even when the
selected command did not use an LLM.

Implemented:

- Converted `simulation/__init__.py` and `learning/__init__.py` to lazy exports.
- Moved Claude-backed imports in `main.py` behind `_ensure_llm_stack()`.
- Removed the runtime import of `SimulationResult` from `trading/ev_calculator.py`.

## Development Roadmap

1. Fix first-run setup and live-trading safety before any real operation.
2. Add unit tests for config loading, CLI setup, EV calculation, live-mode gating, calibration report rendering, and position accounting.
3. Remove tracked generated artifacts.
4. Add a documented dry-run/demo smoke test.
5. Split network clients behind interfaces to make tests deterministic.
6. Add explicit runtime safeguards for real-money trading: max daily loss, max concurrent exposure, startup confirmation, and balance checks.
