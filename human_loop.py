"""
Human-in-the-Loop (HITL) Approval Manager for Albert Miro Weather.

Two modes:

  interactive  — Before each trade Albert pauses, displays a rich approval panel
                 with full signal details (EV, scenarios, Kelly sizing, reasoning),
                 and waits for human input.
                 Best for: --run, --demo, supervised single cycles.

  queue        — Signals are written to pending_review.json instead of executed
                 immediately. Human runs `python main.py --review` at any time to
                 approve or reject them. Approved signals execute on the next cycle.
                 Best for: --daemon, async/overnight operation.

CLI integration (see main.py):
  --human-loop                  interactive mode (default when --human-loop set)
  --human-loop-queue            queue mode
  --hl-min-ev   0.10            only prompt for EV >= this (0 = always prompt)
  --hl-min-usd  0.00            only prompt for size >= this (0 = always prompt)
  --hl-timeout  300             queue mode: wait N seconds for decision before skipping
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from trading.ev_calculator import TradeSignal

_QUEUE_FILE = "pending_review.json"
_console = Console()


# ─── Signal record stored in the queue file ───────────────────────────────────

@dataclass
class QueuedSignal:
    id: str
    queued_at: str
    city: str
    direction: str
    bucket_low: float
    bucket_high: float
    target_date: str
    probability: float
    market_price: float
    ev: float
    kelly_fraction: float
    recommended_usd: float
    confidence_level: str
    hours_to_resolution: float
    volume: float
    market_id: str
    scenarios: list[dict] = field(default_factory=list)
    scenario_reasoning: str = ""
    decision: Optional[str] = None   # None / "approve" / "reject"
    decided_at: Optional[str] = None

    @classmethod
    def from_signal(
        cls,
        signal: "TradeSignal",
        sim_state: Optional[dict] = None,
    ) -> "QueuedSignal":
        city_sim = (sim_state or {}).get(signal.city, {})
        return cls(
            id=f"{signal.city}_{signal.target_date}_{signal.direction}_{int(time.time())}",
            queued_at=datetime.now(timezone.utc).isoformat(),
            city=signal.city,
            direction=signal.direction,
            bucket_low=signal.bucket_low,
            bucket_high=signal.bucket_high,
            target_date=signal.target_date,
            probability=signal.probability,
            market_price=signal.market_price,
            ev=signal.ev,
            kelly_fraction=signal.kelly_fraction,
            recommended_usd=signal.recommended_usd,
            confidence_level=signal.confidence_level,
            hours_to_resolution=signal.hours_to_resolution,
            volume=signal.volume,
            market_id=signal.market_id,
            scenarios=city_sim.get("scenarios", []),
            scenario_reasoning=city_sim.get("scenario_reasoning", ""),
        )


# ─── Rich signal panel ─────────────────────────────────────────────────────────

def _render_signal_panel(qs: QueuedSignal, index: int = 0, total: int = 0) -> Panel:
    """Build a rich approval panel for one signal."""
    lo = f"{qs.bucket_low:.0f}" if qs.bucket_low != float("-inf") else "−∞"
    hi = f"{qs.bucket_high:.0f}" if qs.bucket_high != float("inf") else "+∞"
    bucket = f"{lo}–{hi}°F"

    dir_color = "bold green" if qs.direction == "YES" else "bold red"
    ev_color  = "bold green" if qs.ev >= 0.20 else "green" if qs.ev >= 0.10 else "yellow"
    conf_color = {"high": "bold green", "medium": "yellow", "low": "red"}.get(
        qs.confidence_level, "white"
    )
    counter = f" [{index}/{total}]" if total > 1 else ""

    # ── Header ──
    header = Text()
    header.append("  TRADE SIGNAL — AWAITING APPROVAL", style="bold white")
    header.append(counter, style="dim")

    # ── Details grid ──
    details = Table.grid(padding=(0, 2))
    details.add_column(style="dim white", no_wrap=True)
    details.add_column(style="bold white", no_wrap=True)
    details.add_column(style="dim white", no_wrap=True)
    details.add_column(style="bold white", no_wrap=True)

    details.add_row("City:",      qs.city,            "Date:",       qs.target_date)
    details.add_row("Direction:", Text(qs.direction, style=dir_color),
                    "Bucket:",    bucket)
    details.add_row("Our p:",     f"{qs.probability:.3f}",
                    "Market p:",  f"{qs.market_price:.3f}")
    details.add_row("EV:",        Text(f"{qs.ev:+.4f}", style=ev_color),
                    "Kelly:",     f"{qs.kelly_fraction:.1%}")
    details.add_row("Size:",      Text(f"${qs.recommended_usd:.2f}", style="bold yellow"),
                    "Confidence:", Text(qs.confidence_level.upper(), style=conf_color))
    details.add_row("Volume:",    f"{qs.volume:,.0f}",
                    "Hours left:", f"{qs.hours_to_resolution:.1f}h")

    # ── Scenario block ──
    scenario_lines = Text()
    if qs.scenarios:
        scenario_lines.append("\n  ◆ Scenarios (Morgan)\n", style="bold magenta")
        for sc in qs.scenarios:
            bar_len = int(sc.get("probability", 0) * 14)
            bar = "▓" * bar_len + "░" * (14 - bar_len)
            scenario_lines.append(f"  {bar} {sc.get('probability', 0):.0%}  ", style="magenta")
            scenario_lines.append(f"{sc.get('name', ''):<28}", style="white")
            scenario_lines.append(f"  exp {sc.get('expected_temp_f', 0):.0f}°F\n", style="dim cyan")
        if qs.scenario_reasoning:
            scenario_lines.append(
                f"\n  River: {qs.scenario_reasoning[:120]}\n",
                style="dim italic white",
            )

    body = Text()
    body.append_text(scenario_lines)

    title_color = "green" if qs.direction == "YES" else "red"
    title = f"[{title_color}]{qs.direction}[/]  [bold white]{qs.city}  {bucket}[/]  [dim]{qs.target_date}[/]"

    from rich.console import Group
    return Panel(
        Group(header, Rule(style="dim"), details, body),
        title=title,
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    )


# ─── Manager ──────────────────────────────────────────────────────────────────

class HumanApprovalManager:
    """
    Central controller for human-in-the-loop trade approval.

    Parameters
    ----------
    mode          "interactive" or "queue"
    min_ev        Only ask for approval when signal EV >= this value.
                  Set to 0.0 to always ask.
    min_usd       Only ask when recommended_usd >= this value.
    timeout_secs  Queue mode: seconds to wait for a decision before auto-skipping.
    """

    def __init__(
        self,
        mode: str = "interactive",
        min_ev: float = 0.0,
        min_usd: float = 0.0,
        timeout_secs: int = 300,
    ) -> None:
        if mode not in ("interactive", "queue"):
            raise ValueError(f"mode must be 'interactive' or 'queue', got {mode!r}")
        self.mode = mode
        self.min_ev = min_ev
        self.min_usd = min_usd
        self.timeout_secs = timeout_secs
        self._approve_all = False
        self._skip_all = False

    # ─── Public API ───────────────────────────────────────────────────────────

    def needs_review(self, signal: "TradeSignal") -> bool:
        """Return True if this signal requires human review."""
        if self._approve_all or self._skip_all:
            return False
        return signal.ev >= self.min_ev and signal.recommended_usd >= self.min_usd

    def request_approval(
        self,
        signal: "TradeSignal",
        sim_state: Optional[dict] = None,
    ) -> bool:
        """
        Ask the human whether to execute `signal`.
        Returns True if approved, False if rejected/skipped.
        """
        if self._approve_all:
            return True
        if self._skip_all:
            return False

        qs = QueuedSignal.from_signal(signal, sim_state)

        if self.mode == "interactive":
            return self._interactive(qs)
        else:
            return self._enqueue(qs)

    # ─── Interactive mode ─────────────────────────────────────────────────────

    def _interactive(self, qs: QueuedSignal) -> bool:
        _console.print()
        _console.print(_render_signal_panel(qs))
        _console.print(
            "  [bold cyan][y][/] Approve  "
            "[bold red][n][/] Reject  "
            "[bold yellow][a][/] Approve ALL remaining  "
            "[bold yellow][s][/] Skip ALL remaining  "
            "[bold red][q][/] Quit agent"
        )
        while True:
            try:
                choice = input("  → ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _console.print("\n[yellow]Interrupted — skipping trade.[/]")
                return False

            if choice in ("y", "yes"):
                _console.print(f"  [bold green]✓ Approved:[/] {qs.city} {qs.direction} {qs.recommended_usd:.2f}")
                return True
            elif choice in ("n", "no", ""):
                _console.print(f"  [bold red]✗ Rejected:[/] {qs.city} {qs.direction}")
                return False
            elif choice in ("a", "all"):
                _console.print("  [bold yellow]Approve all remaining signals this cycle.[/]")
                self._approve_all = True
                return True
            elif choice in ("s", "skip"):
                _console.print("  [bold yellow]Skipping all remaining signals this cycle.[/]")
                self._skip_all = True
                return False
            elif choice in ("q", "quit"):
                _console.print("\n[bold red]Quitting agent on user request.[/]")
                raise SystemExit(0)
            else:
                _console.print("  [dim]Unknown input. Use y / n / a / s / q[/]")

    def reset_cycle_flags(self) -> None:
        """Call at the start of each new cycle to reset approve-all / skip-all."""
        self._approve_all = False
        self._skip_all = False

    # ─── Queue mode ───────────────────────────────────────────────────────────

    def _load_queue(self) -> list[dict]:
        if not os.path.exists(_QUEUE_FILE):
            return []
        try:
            with open(_QUEUE_FILE) as f:
                return json.load(f)
        except Exception:
            return []

    def _save_queue(self, queue: list[dict]) -> None:
        with open(_QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

    def _enqueue(self, qs: QueuedSignal) -> bool:
        """
        Write signal to queue and wait up to timeout_secs for a human decision.
        Returns True if approved within the timeout window, False otherwise.
        """
        queue = self._load_queue()
        entry = asdict(qs)
        entry["decision"] = None
        queue.append(entry)
        self._save_queue(queue)

        _console.print(
            f"  [bold yellow]⏳ QUEUED for review:[/] {qs.city} {qs.direction} "
            f"${qs.recommended_usd:.2f} EV={qs.ev:+.3f}  "
            f"[dim](run `python main.py --review` to approve)[/]"
        )

        # Poll for decision
        deadline = time.time() + self.timeout_secs
        while time.time() < deadline:
            time.sleep(5)
            queue = self._load_queue()
            for item in queue:
                if item.get("id") == qs.id:
                    if item.get("decision") == "approve":
                        _console.print(f"  [green]✓ Queue approved:[/] {qs.city} {qs.direction}")
                        # Remove from queue after execution
                        queue = [i for i in queue if i.get("id") != qs.id]
                        self._save_queue(queue)
                        return True
                    elif item.get("decision") == "reject":
                        _console.print(f"  [red]✗ Queue rejected:[/] {qs.city} {qs.direction}")
                        queue = [i for i in queue if i.get("id") != qs.id]
                        self._save_queue(queue)
                        return False

        # Timeout — remove and skip
        queue = self._load_queue()
        queue = [i for i in queue if i.get("id") != qs.id]
        self._save_queue(queue)
        _console.print(
            f"  [dim yellow]⏰ Timeout ({self.timeout_secs}s) — skipped {qs.city} {qs.direction}[/]"
        )
        return False

    # ─── --review command ─────────────────────────────────────────────────────

    @staticmethod
    def run_review_cli() -> None:
        """
        Interactive CLI for reviewing pending signals in the queue.
        Called by `python main.py --review`.
        """
        queue = []
        if os.path.exists(_QUEUE_FILE):
            try:
                with open(_QUEUE_FILE) as f:
                    queue = json.load(f)
            except Exception as e:
                _console.print(f"[red]Could not load queue: {e}[/]")
                return

        pending = [q for q in queue if q.get("decision") is None]

        if not pending:
            _console.print("\n[dim]No pending signals in queue.[/]")
            if queue:
                decided = [q for q in queue if q.get("decision") is not None]
                _console.print(f"[dim]{len(decided)} already decided signals in {_QUEUE_FILE}[/]")
            return

        _console.print(f"\n[bold cyan]{len(pending)} signal(s) awaiting your review[/]\n")

        for i, item in enumerate(pending, 1):
            # Reconstruct a QueuedSignal-like object for display
            qs = QueuedSignal(**{k: item.get(k, v.default if hasattr(v, 'default') else None)
                                 for k, v in QueuedSignal.__dataclass_fields__.items()})
            _console.print(_render_signal_panel(qs, index=i, total=len(pending)))
            _console.print(
                "  [bold cyan][y][/] Approve  "
                "[bold red][n][/] Reject  "
                "[bold yellow][a][/] Approve ALL  "
                "[bold yellow][r][/] Reject ALL  "
                "[dim][s][/] Skip (decide later)"
            )

            approved_all = False
            rejected_all = False

            while True:
                try:
                    choice = input("  → ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    _console.print("\n[yellow]Review interrupted.[/]")
                    break

                if choice in ("y", "yes"):
                    item["decision"] = "approve"
                    item["decided_at"] = datetime.now(timezone.utc).isoformat()
                    _console.print(f"  [green]✓ Approved[/]")
                    break
                elif choice in ("n", "no"):
                    item["decision"] = "reject"
                    item["decided_at"] = datetime.now(timezone.utc).isoformat()
                    _console.print(f"  [red]✗ Rejected[/]")
                    break
                elif choice in ("a",):
                    approved_all = True
                    item["decision"] = "approve"
                    item["decided_at"] = datetime.now(timezone.utc).isoformat()
                    _console.print("[yellow]Approving all remaining...[/]")
                    break
                elif choice in ("r",):
                    rejected_all = True
                    item["decision"] = "reject"
                    item["decided_at"] = datetime.now(timezone.utc).isoformat()
                    _console.print("[yellow]Rejecting all remaining...[/]")
                    break
                elif choice in ("s", "skip", ""):
                    _console.print("  [dim]Skipped — will ask again next time.[/]")
                    break
                else:
                    _console.print("  [dim]Use y / n / a / r / s[/]")

            if approved_all:
                for remaining in pending[i:]:
                    remaining["decision"] = "approve"
                    remaining["decided_at"] = datetime.now(timezone.utc).isoformat()
                break
            if rejected_all:
                for remaining in pending[i:]:
                    remaining["decision"] = "reject"
                    remaining["decided_at"] = datetime.now(timezone.utc).isoformat()
                break

        # Save updated queue
        with open(_QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

        approved  = sum(1 for q in pending if q.get("decision") == "approve")
        rejected  = sum(1 for q in pending if q.get("decision") == "reject")
        undecided = sum(1 for q in pending if q.get("decision") is None)
        _console.print(
            f"\n[bold]Review complete:[/] "
            f"[green]{approved} approved[/]  "
            f"[red]{rejected} rejected[/]  "
            f"[dim]{undecided} skipped[/]"
        )

    @staticmethod
    def show_queue_status() -> None:
        """Print a summary of what's currently in the queue."""
        if not os.path.exists(_QUEUE_FILE):
            _console.print("[dim]Queue file does not exist.[/]")
            return
        try:
            with open(_QUEUE_FILE) as f:
                queue = json.load(f)
        except Exception:
            _console.print("[red]Could not read queue.[/]")
            return

        if not queue:
            _console.print("[dim]Queue is empty.[/]")
            return

        table = Table(title="Pending Review Queue", box=box.ROUNDED, show_header=True)
        table.add_column("ID", style="dim", max_width=30)
        table.add_column("City")
        table.add_column("Dir")
        table.add_column("EV")
        table.add_column("$Size")
        table.add_column("Status")

        for item in queue:
            decision = item.get("decision")
            status_text = {
                None: Text("pending", style="yellow"),
                "approve": Text("approved", style="green"),
                "reject": Text("rejected", style="red"),
            }.get(decision, Text(str(decision), style="dim"))

            ev = item.get("ev", 0)
            ev_style = "green" if ev >= 0.10 else "yellow"
            table.add_row(
                item.get("id", "")[:28],
                item.get("city", ""),
                Text(item.get("direction", ""), style="green" if item.get("direction") == "YES" else "red"),
                Text(f"{ev:+.3f}", style=ev_style),
                f"${item.get('recommended_usd', 0):.2f}",
                status_text,
            )

        _console.print(table)
