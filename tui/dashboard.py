"""
Albert Miro Weather — Terminal Dashboard

Real-time TUI dashboard powered by `rich`.  Launched via `python main.py --tui`.
Layout mirrors the Hermes Weather Agent style:
  header  │ large banner + session info
  left    │ agent status, learning metrics, open positions
  center  │ live log stream
  right   │ city forecast chart (top) + market scan table (bottom)
  footer  │ live city ticker
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from main import MiroWeatherAgent
    from .log_handler import TuiLogHandler

VERSION = "v1.0"

# ASCII banner — displayed in header
_BANNER_LINE1 = "▄▀█ █   █▄▄ █▀▀ █▀█ ▀█▀   █▀▄▀█ █ █▀█ █▀█   █░█░█ █▀▀ ▄▀█ ▀█▀ █ █ █▀▀ █▀█"
_BANNER_LINE2 = "█▀█ █▄▄ █▄█ ██▄ █▀▄  █    █░▀░█ █ █▀▄ █▄█   ▀▄▀▄▀ ██▄ █▀█  █  █▀█ ██▄ █▀▄"


class AlbertDashboard:
    """
    Renders the full Albert Miro Weather TUI.
    The agent runs in a daemon thread; this class owns the main thread
    and calls rich.live.Live.
    """

    def __init__(self, agent: "MiroWeatherAgent", log_handler: "TuiLogHandler") -> None:
        self._agent = agent
        self._log = log_handler
        self._console = Console()
        self._start_time = datetime.now(timezone.utc)

    # ─── Public API ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block the calling thread, rendering the TUI until Ctrl+C."""
        layout = self._make_layout()
        with Live(
            layout,
            console=self._console,
            refresh_per_second=2,
            screen=True,
        ):
            try:
                while True:
                    self._update(layout)
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    # ─── Layout skeleton ──────────────────────────────────────────────────────

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=22),
            Layout(name="center", ratio=44),
            Layout(name="right", ratio=34),
        )
        layout["right"].split_column(
            Layout(name="chart", ratio=42),
            Layout(name="scan", ratio=58),
        )
        return layout

    def _update(self, layout: Layout) -> None:
        layout["header"].update(self._render_header())
        layout["left"].update(self._render_left())
        layout["center"].update(self._render_center())
        layout["chart"].update(self._render_chart())
        layout["scan"].update(self._render_scan())
        layout["footer"].update(self._render_footer())

    # ─── Section renderers ────────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        now = datetime.now(timezone.utc)
        elapsed = int((now - self._start_time).total_seconds())
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        state = self._state()

        banner = Text(justify="center")
        banner.append(_BANNER_LINE1 + "\n", style="bold cyan")
        banner.append(_BANNER_LINE2 + "\n", style="bold cyan")

        meta = Text(justify="center")
        meta.append(f"● Albert Miro Weather  {VERSION}  ({now.strftime('%Y-%m-%d')})", style="bold white")
        meta.append(f"  ●  uptime {h:02d}:{m:02d}:{s:02d}", style="dim green")
        meta.append(f"  ●  cycle #{state.get('cycle_num', 0)}", style="dim yellow")
        meta.append(f"  ●  {now.strftime('%H:%M:%S')} UTC", style="dim white")
        meta.append(f"  ●  {state.get('current_city', '—')}", style="dim magenta")

        return Panel(
            Group(banner, meta),
            style="bold cyan",
            box=box.DOUBLE_EDGE,
        )

    def _render_left(self) -> Panel:
        agent = self._agent
        pos_summary = agent.positions.summary()
        stats = agent.memory.overall_stats()

        deployed = sum(p.size_usd for p in agent.positions.open_positions.values())

        # Status grid
        status = Table.grid(padding=(0, 1))
        status.add_column(style="dim white", no_wrap=True)
        status.add_column(style="bold white", no_wrap=True)
        status.add_row("Positions:", str(pos_summary["open_positions"]) + " open")
        status.add_row("Deployed:", f"${deployed:.2f}")
        win = pos_summary["win_rate"]
        status.add_row("Win rate:", f"{win:.1%}" if win else "N/A")
        pnl = pos_summary["total_pnl_usd"]
        pnl_markup = f"[green]+${pnl:.2f}[/]" if pnl >= 0 else f"[red]-${abs(pnl):.2f}[/]"
        status.add_row("Total PnL:", pnl_markup)

        state = self._state()
        next_scan_mins = max(0, 60 - (int((datetime.now(timezone.utc) - self._start_time).total_seconds()) % 3600) // 60)
        status.add_row("Next scan:", f"+{next_scan_mins} min")

        # Learning grid
        learn = Table.grid(padding=(0, 1))
        learn.add_column(style="dim white", no_wrap=True)
        learn.add_column(style="bold cyan", no_wrap=True)
        learn.add_row("Lessons:", str(stats.get("total_lessons", 0)))
        learn.add_row("Predictions:", str(stats.get("total_predictions", 0)))
        brier = agent.calibrator.brier_score()
        learn.add_row("Brier:", f"{brier:.4f}" if brier else "N/A")
        ece = agent.calibrator.expected_calibration_error()
        learn.add_row("ECE:", f"{ece:.4f}" if ece else "N/A")

        # Overconfidence flags
        flags = agent.calibrator.city_overconfidence_flags()
        flag_lines = Text()
        if flags:
            for f in flags[:3]:
                arrow = "▲" if f["direction"] == "overconfident" else "▼"
                flag_lines.append(f" {arrow} {f['city']}: {f['mean_calibration_error']:+.2f}\n",
                                  style="bold yellow")
        else:
            flag_lines.append(" ✓ Calibration nominal\n", style="dim green")

        # Open positions table
        pos_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            padding=(0, 0),
            expand=True,
        )
        pos_table.add_column("City", max_width=9, no_wrap=True)
        pos_table.add_column("Dir", no_wrap=True)
        pos_table.add_column("Entry", no_wrap=True)
        pos_table.add_column("PnL%", no_wrap=True)

        for pos in list(agent.positions.open_positions.values())[:10]:
            dir_style = "green" if pos.direction == "YES" else "red"
            pnl_pct = pos.unrealized_pnl_pct
            pnl_style = "green" if pnl_pct >= 0 else "red"
            pos_table.add_row(
                pos.city[:9],
                Text(pos.direction, style=dir_style),
                f"{pos.entry_price:.3f}",
                Text(f"{pnl_pct:+.1%}", style=pnl_style),
            )

        content = Group(
            Text("● Status", style="bold white"),
            status,
            Rule(style="dim"),
            Text("● Learning", style="bold cyan"),
            learn,
            flag_lines,
            Rule(style="dim"),
            Text("● Open Positions", style="bold white"),
            pos_table if agent.positions.open_positions
            else Text(" No open positions", style="dim"),
        )

        return Panel(
            content,
            title="[bold cyan]albert-miro-v1[/]",
            style="cyan dim",
            box=box.ROUNDED,
        )

    def _render_center(self) -> Panel:
        entries = list(self._log.buffer)[-50:]
        content = Text()
        for entry in entries:
            content.append_text(entry)
            content.append("\n")

        state = self._state()
        title = (
            f"[bold white]Agent Log[/]  "
            f"[dim]streaming • turn {len(self._log.buffer)}[/]"
        )
        return Panel(content, title=title, style="white dim", box=box.ROUNDED)

    def _render_chart(self) -> Panel:
        state = self._state()
        forecasts = state.get("last_forecast", {})
        city = state.get("current_city", "")

        if not city or city not in forecasts:
            city = next(iter(forecasts), "")

        if not city:
            return Panel(
                Align.center(Text("Waiting for data…", style="dim")),
                title="[bold white]Forecast vs Polymarket[/]",
                style="blue dim",
                box=box.ROUNDED,
            )

        forecast_f = forecasts.get(city, 0.0)
        markets = state.get("last_markets", {}).get(city, [])

        lines = Text()
        lines.append(f" {city}", style="bold white")
        lines.append(f"  —  forecast: {forecast_f:.1f}°F\n\n", style="dim white")

        if markets:
            for m in markets[:6]:
                lo = m["bucket_low"]
                hi = m["bucket_high"]
                lo_str = f"{lo:.0f}" if lo != float("-inf") else "−∞"
                hi_str = f"{hi:.0f}" if hi != float("inf") else "+∞"
                bucket = f"{lo_str}–{hi_str}°F"
                p = m["price_yes"]
                bar_len = max(1, int(p * 22))
                bar = "█" * bar_len + "░" * (22 - bar_len)
                p_style = "bold green" if p > 0.55 else "bold red" if p < 0.35 else "yellow"
                lines.append(f" {bucket:<14}", style="dim white")
                lines.append(f"{bar}", style=p_style)
                lines.append(f" {p:.2f}\n", style="white")
        else:
            lines.append(" No markets found\n", style="dim yellow")

        # Signals for this city
        signals = [s for s in state.get("last_signals", []) if s.city == city]
        if signals:
            lines.append("\n Signals:\n", style="bold dim")
            for sig in signals[:3]:
                ev_s = "bold green" if sig.ev >= 0.15 else "green" if sig.ev > 0 else "red"
                lo_str = f"{sig.bucket_low:.0f}" if sig.bucket_low != float("-inf") else "−∞"
                hi_str = f"{sig.bucket_high:.0f}" if sig.bucket_high != float("inf") else "+∞"
                lines.append(f"  {sig.direction} {lo_str}–{hi_str}°F ", style="bold white")
                lines.append(f"EV={sig.ev:+.3f} ", style=ev_s)
                lines.append(f"p={sig.probability:.2f} ", style="cyan")
                lines.append(f"${sig.recommended_usd:.2f}\n", style="yellow")

        return Panel(lines, title=f"[bold white]{city}[/]", style="blue dim", box=box.ROUNDED)

    def _render_scan(self) -> Panel:
        state = self._state()
        all_signals = state.get("last_signals", [])
        forecasts = state.get("last_forecast", {})

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold dim",
            padding=(0, 0),
            expand=True,
        )
        table.add_column("City", max_width=12, no_wrap=True, style="white")
        table.add_column("Fcst", no_wrap=True)
        table.add_column("PM", no_wrap=True)
        table.add_column("EV", no_wrap=True)

        seen: set[str] = set()
        for sig in sorted(all_signals, key=lambda s: -abs(s.ev)):
            if sig.city in seen:
                continue
            seen.add(sig.city)
            ev_s = "bold green" if sig.ev >= 0.15 else "green" if sig.ev > 0 else "red"
            fcst = forecasts.get(sig.city)
            fcst_str = f"{fcst:.1f}°F" if fcst else f"{sig.probability:.0%}"
            table.add_row(
                sig.city[:12],
                fcst_str,
                f"{sig.market_price:.2f}",
                Text(f"{sig.ev:+.2f}", style=ev_s),
            )

        # Fill remaining cities without signals
        for city, temp in forecasts.items():
            if city in seen:
                continue
            table.add_row(city[:12], f"{temp:.1f}°F", "—", Text("—", style="dim"))
            seen.add(city)
            if len(seen) >= 20:
                break

        n_markets = sum(len(v) for v in state.get("last_markets", {}).values())
        n_edges = len(all_signals)
        title = f"[bold white]Market Scan[/]  [dim]{n_markets} markets • {n_edges} edges[/]"

        return Panel(table, title=title, style="blue dim", box=box.ROUNDED)

    def _render_footer(self) -> Panel:
        state = self._state()
        signals = state.get("last_signals", [])
        forecasts = state.get("last_forecast", {})

        ticker = Text(overflow="ellipsis", no_wrap=True)
        ticker.append("▶ LIVE  ", style="bold green")

        shown: set[str] = set()
        for sig in sorted(signals, key=lambda s: -abs(s.ev))[:12]:
            if sig.city in shown:
                continue
            shown.add(sig.city)
            fcst = forecasts.get(sig.city)
            temp_str = f"{fcst:.0f}°F" if fcst else f"{sig.probability:.0%}"
            ev_s = "bold green" if sig.ev >= 0.15 else "green" if sig.ev > 0 else "red"
            ticker.append(f"{sig.city} {temp_str} ", style="bold white")
            ticker.append(f"EV{sig.ev:+.1f}", style=ev_s)
            ticker.append("  •  ", style="dim")

        for city, temp in forecasts.items():
            if city in shown:
                continue
            ticker.append(f"{city} {temp:.0f}°F  •  ", style="dim white")

        return Panel(ticker, style="green dim", box=box.ROUNDED, height=3)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _state(self) -> dict:
        return getattr(self._agent, "_cycle_state", {})
