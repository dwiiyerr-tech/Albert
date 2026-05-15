"""
Demo / Paper-Trading Session for Albert Miro Weather.

Provides a safe sandbox to validate the agent works correctly before
risking real funds.  Features:
  • Virtual wallet  — configurable starting balance, blocks overspending
  • Token tracking  — estimates all LLM tokens + USD cost
  • Error collector — captures every exception with context; never crashes
  • Cycle budgets   — hard stop after N cycles
  • Session report  — full rich-formatted summary at exit

Usage (from main.py --demo):
    session = DemoSession(virtual_balance=1000.0, max_cycles=3)
    session.install_token_tracking()   # call BEFORE LLM clients are created
    agent = MiroWeatherAgent(demo_session=session)
    agent.run_demo(...)
    session.print_report()
"""
from __future__ import annotations

import datetime
import traceback
from dataclasses import dataclass, field
from typing import Optional

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:
    import re

    def _plain(value: object) -> str:
        return re.sub(r"\[/?[^\]]+\]", "", str(value))

    class _Box:
        ROUNDED = None
        SIMPLE_HEAVY = None
        DOUBLE_EDGE = None

    box = _Box()

    class Console:
        def print(self, *objects, **kwargs) -> None:
            print(*(_plain(obj) for obj in objects))

        def rule(self, title: str = "", **kwargs) -> None:
            print(f"\n{'-' * 8} {_plain(title)} {'-' * 8}")

    class Group:
        def __init__(self, *items) -> None:
            self.items = items

        def __str__(self) -> str:
            return "\n".join(_plain(i) for i in self.items)

    class Panel:
        def __init__(self, renderable, *args, title: str = "", **kwargs) -> None:
            self.renderable = renderable
            self.title = title

        def __str__(self) -> str:
            title = _plain(self.title)
            body = _plain(self.renderable)
            return f"{title}\n{body}" if title else body

    class Rule:
        def __init__(self, title: str = "", **kwargs) -> None:
            self.title = title

        def __str__(self) -> str:
            return f"{'-' * 8} {_plain(self.title)} {'-' * 8}"

    class Table:
        def __init__(self, *args, **kwargs) -> None:
            self._rows: list[tuple[str, ...]] = []

        @classmethod
        def grid(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def add_column(self, *args, **kwargs) -> None:
            return None

        def add_row(self, *values) -> None:
            self._rows.append(tuple(_plain(v) for v in values))

        def __str__(self) -> str:
            return "\n".join("  ".join(row) for row in self._rows)

    class Text:
        def __init__(self, text: str = "", *args, **kwargs) -> None:
            self._parts = [text] if text else []

        def append(self, text: str, *args, **kwargs) -> None:
            self._parts.append(text)

        def __str__(self) -> str:
            return "".join(self._parts)

# ─── Default pricing estimate (USD per token, approximate) ───────────────────
_INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000   # $3.00 / M input tokens
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000   # $15.00 / M output tokens

# ─── Token budget warning threshold ─────────────────────────────────────────
DEFAULT_TOKEN_BUDGET = 200_000   # warn (not block) above this many input tokens


# ─────────────────────────────────────────────────────────────────────────────

class TokenCounter:
    """Tracks cumulative token usage across all LLM calls."""

    def __init__(self, budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.api_calls: int = 0
        self.budget: int = budget
        self._warned: bool = False

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.api_calls += 1
        if not self._warned and self.input_tokens > self.budget:
            self._warned = True
            print(
                f"\n⚠  DEMO TOKEN BUDGET WARNING: {self.input_tokens:,} input tokens used "
                f"(budget: {self.budget:,}).  Consider reducing --demo-cycles or "
                f"--demo-sim-rounds to save API costs.\n"
            )

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens  * _INPUT_COST_PER_TOKEN
            + self.output_tokens * _OUTPUT_COST_PER_TOKEN
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ─────────────────────────────────────────────────────────────────────────────

class VirtualWallet:
    """
    Simulates a USDC wallet for paper trading.
    Balance = start_balance - deployed_capital + closed_pnl.
    call can_open() before opening a position.
    """

    def __init__(self, balance: float = 1000.0) -> None:
        self.start_balance: float = balance
        self._deposits: float = 0.0
        self._withdrawals: float = 0.0

    def available(self, position_manager) -> float:
        """Current spendable balance (start + closed PnL - open deployed)."""
        deployed = sum(p.size_usd for p in position_manager.open_positions.values())
        return self.start_balance + position_manager.total_pnl() - deployed

    def can_open(self, amount: float, position_manager) -> bool:
        return self.available(position_manager) >= amount

    @property
    def net_pnl_desc(self) -> str:
        return "N/A (requires PositionManager)"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ErrorRecord:
    cycle: int
    context: str
    exc_type: str
    message: str
    tb: str


@dataclass
class _TradeRecord:
    cycle: int
    city: str
    direction: str
    bucket: str
    ev: float
    probability: float
    market_price: float
    recommended_usd: float
    executed: bool
    skip_reason: str = ""


@dataclass
class _CycleSummary:
    cycle: int
    started_at: str
    duration_seconds: float
    signals_found: int
    trades_executed: int
    errors: int
    input_tokens_this_cycle: int
    output_tokens_this_cycle: int


# ─────────────────────────────────────────────────────────────────────────────

class DemoSession:
    """
    Manages the full lifecycle of a demo/paper-trading session.
    Must call install_token_tracking() before LLM clients are created.
    """

    def __init__(
        self,
        virtual_balance: float = 1000.0,
        max_cycles: int = 3,
        cycle_interval_seconds: int = 0,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.virtual_balance = virtual_balance
        self.max_cycles = max_cycles
        self.cycle_interval_seconds = cycle_interval_seconds

        self.wallet = VirtualWallet(virtual_balance)
        self.tokens = TokenCounter(token_budget)

        self._errors: list[_ErrorRecord] = []
        self._trades: list[_TradeRecord] = []
        self._cycles: list[_CycleSummary] = []

        self._cycle_start_time: Optional[datetime.datetime] = None
        self._cycle_tokens_start: int = 0
        self._cycle_output_start: int = 0
        self.current_cycle: int = 0
        self._tracking_installed: bool = False

    # ─── Token tracking ───────────────────────────────────────────────────────

    def install_token_tracking(self) -> None:
        """
        Monkey-patch the provider-neutral LLMClient so every call is counted.
        """
        if self._tracking_installed:
            return

        _tracker = self.tokens

        try:
            from llm_client import LLMClient
            _orig_text = LLMClient.text

            def _tracked_text(self_client, *args, **kwargs):
                system = kwargs.get("system", "")
                messages = kwargs.get("messages", [])
                approx_in = len(str(system)) // 4
                approx_in += sum(len(str(m.get("content", ""))) // 4 for m in messages)
                response_text = _orig_text(self_client, *args, **kwargs)
                approx_out = len(response_text) // 4
                _tracker.add(max(1, approx_in), max(1, approx_out))
                return response_text

            LLMClient.text = _tracked_text
        except Exception:
            pass

        self._tracking_installed = True

    # ─── Cycle lifecycle ─────────────────────────────────────────────────────

    def begin_cycle(self, cycle_num: int) -> None:
        self.current_cycle = cycle_num
        self._cycle_start_time = datetime.datetime.now()
        self._cycle_tokens_start = self.tokens.input_tokens
        self._cycle_output_start = self.tokens.output_tokens

    def end_cycle(self, signals_found: int, trades_executed: int) -> None:
        duration = (
            (datetime.datetime.now() - self._cycle_start_time).total_seconds()
            if self._cycle_start_time else 0.0
        )
        errors_this_cycle = sum(1 for e in self._errors if e.cycle == self.current_cycle)
        self._cycles.append(_CycleSummary(
            cycle=self.current_cycle,
            started_at=self._cycle_start_time.isoformat() if self._cycle_start_time else "",
            duration_seconds=round(duration, 1),
            signals_found=signals_found,
            trades_executed=trades_executed,
            errors=errors_this_cycle,
            input_tokens_this_cycle=self.tokens.input_tokens - self._cycle_tokens_start,
            output_tokens_this_cycle=self.tokens.output_tokens - self._cycle_output_start,
        ))

    @property
    def cycles_completed(self) -> int:
        return len(self._cycles)

    def should_continue(self) -> bool:
        return self.cycles_completed < self.max_cycles

    # ─── Trade tracking ──────────────────────────────────────────────────────

    def record_trade(
        self,
        signal,
        executed: bool,
        skip_reason: str = "",
    ) -> None:
        lo = signal.bucket_low
        hi = signal.bucket_high
        lo_s = f"{lo:.0f}" if lo != float("-inf") else "−∞"
        hi_s = f"{hi:.0f}" if hi != float("inf") else "+∞"
        self._trades.append(_TradeRecord(
            cycle=self.current_cycle,
            city=signal.city,
            direction=signal.direction,
            bucket=f"{lo_s}–{hi_s}°F",
            ev=signal.ev,
            probability=signal.probability,
            market_price=signal.market_price,
            recommended_usd=signal.recommended_usd,
            executed=executed,
            skip_reason=skip_reason,
        ))

    # ─── Error collection ────────────────────────────────────────────────────

    def record_error(self, context: str, exc: Exception) -> None:
        self._errors.append(_ErrorRecord(
            cycle=self.current_cycle,
            context=context,
            exc_type=type(exc).__name__,
            message=str(exc),
            tb=traceback.format_exc(),
        ))

    # ─── Session report ──────────────────────────────────────────────────────

    def print_report(self, position_manager=None) -> None:
        console = Console()
        console.print()
        console.rule("[bold cyan]  ALBERT MIRO WEATHER — DEMO SESSION REPORT  ", style="cyan")
        console.print()

        # ── Overview ─────────────────────────────────────────────────────────
        total_dur = sum(c.duration_seconds for c in self._cycles)
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="dim white")
        overview.add_column(style="bold white")
        overview.add_row("Cycles completed:", str(len(self._cycles)))
        overview.add_row("Total wall time:", f"{total_dur:.1f}s")
        overview.add_row("Signals found:", str(sum(c.signals_found for c in self._cycles)))
        overview.add_row("Trades executed:", str(sum(c.trades_executed for c in self._cycles)))
        overview.add_row("Errors caught:", str(len(self._errors)))
        console.print(Panel(overview, title="[bold white]● Overview[/]", box=box.ROUNDED))

        # ── Token usage ──────────────────────────────────────────────────────
        tok = self.tokens
        tok_table = Table.grid(padding=(0, 2))
        tok_table.add_column(style="dim white")
        tok_table.add_column(style="bold cyan")
        tok_table.add_row("API calls:", str(tok.api_calls))
        tok_table.add_row("Input tokens:", f"{tok.input_tokens:,}")
        tok_table.add_row("Output tokens:", f"{tok.output_tokens:,}")
        tok_table.add_row("Total tokens:", f"{tok.total_tokens:,}")
        cost = tok.cost_usd
        cost_style = "bold red" if cost > 0.50 else "bold green"
        tok_table.add_row("Estimated cost:", Text(f"${cost:.4f} USD", style=cost_style))
        console.print(Panel(tok_table, title="[bold white]● API Token Usage[/]", box=box.ROUNDED))

        # ── Virtual wallet ────────────────────────────────────────────────────
        if position_manager:
            deployed = sum(p.size_usd for p in position_manager.open_positions.values())
            closed_pnl = position_manager.total_pnl()
            available = self.wallet.start_balance + closed_pnl - deployed
            wr = position_manager.win_rate()

            wallet_table = Table.grid(padding=(0, 2))
            wallet_table.add_column(style="dim white")
            wallet_table.add_column(style="bold white")
            wallet_table.add_row("Starting balance:", f"${self.wallet.start_balance:.2f}")
            wallet_table.add_row("Deployed (open):", f"${deployed:.2f}")
            pnl_s = "bold green" if closed_pnl >= 0 else "bold red"
            wallet_table.add_row("Closed P&L:", Text(f"${closed_pnl:+.2f}", style=pnl_s))
            wallet_table.add_row("Available balance:", f"${available:.2f}")
            wallet_table.add_row("Win rate:", f"{wr:.1%}" if wr else "N/A")
            console.print(Panel(wallet_table, title="[bold white]● Virtual Wallet[/]", box=box.ROUNDED))

        # ── Cycle breakdown ───────────────────────────────────────────────────
        cyc_table = Table(box=box.SIMPLE_HEAVY, show_header=True,
                          header_style="bold dim", expand=True)
        cyc_table.add_column("#", style="dim")
        cyc_table.add_column("Duration")
        cyc_table.add_column("Signals")
        cyc_table.add_column("Trades")
        cyc_table.add_column("Errors")
        cyc_table.add_column("Tokens (in/out)")
        cyc_table.add_column("Est. Cost")
        for c in self._cycles:
            err_s = "bold red" if c.errors else "dim"
            cost_c = (c.input_tokens_this_cycle * _INPUT_COST_PER_TOKEN
                      + c.output_tokens_this_cycle * _OUTPUT_COST_PER_TOKEN)
            cyc_table.add_row(
                str(c.cycle),
                f"{c.duration_seconds:.1f}s",
                str(c.signals_found),
                str(c.trades_executed),
                Text(str(c.errors), style=err_s),
                f"{c.input_tokens_this_cycle:,} / {c.output_tokens_this_cycle:,}",
                f"${cost_c:.4f}",
            )
        console.print(Panel(cyc_table, title="[bold white]● Cycle Breakdown[/]", box=box.ROUNDED))

        # ── Trade log ─────────────────────────────────────────────────────────
        if self._trades:
            trade_table = Table(box=box.SIMPLE_HEAVY, show_header=True,
                                header_style="bold dim", expand=True)
            trade_table.add_column("Cy", style="dim")
            trade_table.add_column("City")
            trade_table.add_column("Dir")
            trade_table.add_column("Bucket")
            trade_table.add_column("EV")
            trade_table.add_column("p")
            trade_table.add_column("$")
            trade_table.add_column("Status")
            for t in self._trades:
                ev_s = "green" if t.ev >= 0.10 else "yellow"
                dir_s = "bold green" if t.direction == "YES" else "bold red"
                status = (
                    Text("✓ executed", style="green") if t.executed
                    else Text(f"✗ {t.skip_reason}", style="dim yellow")
                )
                trade_table.add_row(
                    str(t.cycle),
                    t.city,
                    Text(t.direction, style=dir_s),
                    t.bucket,
                    Text(f"{t.ev:+.3f}", style=ev_s),
                    f"{t.probability:.2f}",
                    f"${t.recommended_usd:.2f}",
                    status,
                )
            console.print(Panel(trade_table, title="[bold white]● Trade Log[/]", box=box.ROUNDED))

        # ── Errors ────────────────────────────────────────────────────────────
        if self._errors:
            console.print(Rule("[bold red]  Errors Caught During Session  ", style="red"))
            for i, err in enumerate(self._errors, 1):
                err_text = Text()
                err_text.append(f"[{i}] Cycle {err.cycle} — {err.context}\n", style="bold red")
                err_text.append(f"    {err.exc_type}: {err.message}\n", style="red")
                if len(err.tb) < 800:
                    err_text.append(err.tb, style="dim red")
                console.print(err_text)
        else:
            console.print(Text("  ✓ No errors during session", style="bold green"))

        # ── Health verdict ────────────────────────────────────────────────────
        console.print()
        if not self._errors and self.cycles_completed >= self.max_cycles:
            verdict = Text(
                "  ✓ DEMO PASSED — agent completed all cycles without errors. "
                "Safe to run in daemon mode.", style="bold green"
            )
        elif self._errors and len(self._errors) <= 2:
            verdict = Text(
                f"  ⚠  DEMO PASSED WITH WARNINGS — {len(self._errors)} error(s) caught. "
                "Review errors above before live deployment.", style="bold yellow"
            )
        else:
            verdict = Text(
                f"  ✗ DEMO NEEDS ATTENTION — {len(self._errors)} error(s) caught. "
                "Fix issues before running with real funds.", style="bold red"
            )
        console.print(Panel(verdict, style="bold", box=box.DOUBLE_EDGE))
        console.print()
