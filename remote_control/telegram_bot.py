"""Telegram remote control for Albert.

The command layer is intentionally provider-neutral enough to unit test without
network access. Telegram polling is a thin adapter around it.
"""
from __future__ import annotations

import contextlib
import datetime
import html
import io
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

SAFE_DEFAULT_COMMANDS = {
    "status",
    "positions",
    "signals",
    "learning",
    "pnl",
    "pause",
    "resume",
    "dry_run_once",
    "demo_once",
}
PUBLIC_COMMANDS = {"start", "help", "whoami"}
ALIASES = {
    "start": "help",
    "pos": "positions",
    "position": "positions",
    "last_signals": "signals",
    "dry": "dry_run_once",
    "run": "dry_run_once",
    "demo": "demo_once",
}
LIVE_COMMANDS = {"live_run_once", "enable_live", "open_trade"}


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _csv_set(raw: str | Iterable[str]) -> set[str]:
    if isinstance(raw, str):
        items = raw.split(",")
    else:
        items = raw
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _format_bucket(low: float, high: float) -> str:
    left = "-inf" if low == float("-inf") else f"{low:g}"
    right = "+inf" if high == float("inf") else f"{high:g}"
    return f"{left}-{right}F"


def _trim_message(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 80].rstrip() + "\n\n[truncated: message too long]"


def _as_telegram_pre(text: str) -> str:
    return f"<pre>{html.escape(_trim_message(text, 3500))}</pre>"


def _signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def _money(value: float) -> str:
    return f"${value:.2f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _table(title: str, rows: list[tuple[str, object]]) -> list[str]:
    width = max([len(label) for label, _value in rows] + [1])
    lines = [title, "-" * len(title)]
    for label, value in rows:
        lines.append(f"{label:<{width}} : {value}")
    return lines


@dataclass
class RemoteControlPolicy:
    allowed_chat_ids: set[str] = field(default_factory=set)
    allowed_commands: set[str] = field(default_factory=lambda: set(SAFE_DEFAULT_COMMANDS))
    allow_live: bool = False
    audit_log: str = "remote_control.log"

    @classmethod
    def from_strings(
        cls,
        *,
        allowed_chat_ids: str,
        allowed_commands: str,
        allow_live: bool = False,
        audit_log: str = "remote_control.log",
    ) -> "RemoteControlPolicy":
        return cls(
            allowed_chat_ids=_csv_set(allowed_chat_ids),
            allowed_commands=_csv_set(allowed_commands) or set(SAFE_DEFAULT_COMMANDS),
            allow_live=allow_live,
            audit_log=audit_log,
        )

    def is_chat_allowed(self, chat_id: int | str) -> bool:
        return str(chat_id) in self.allowed_chat_ids

    def is_command_allowed(self, command: str) -> bool:
        command = canonical_command(command)
        if command in PUBLIC_COMMANDS:
            return True
        if command in LIVE_COMMANDS and not self.allow_live:
            return False
        return command in self.allowed_commands


@dataclass
class RemoteNotificationSettings:
    notification_chat_ids: set[str] = field(default_factory=set)
    notify_cycle_summary: bool = True
    notify_errors: bool = True
    daily_pnl_enabled: bool = True
    pnl_report_interval_hours: float = 24.0
    pnl_report_on_start: bool = False

    @classmethod
    def from_strings(
        cls,
        *,
        notification_chat_ids: str = "",
        notify_cycle_summary: bool = True,
        notify_errors: bool = True,
        daily_pnl_enabled: bool = True,
        pnl_report_interval_hours: float = 24.0,
        pnl_report_on_start: bool = False,
    ) -> "RemoteNotificationSettings":
        return cls(
            notification_chat_ids={str(item) for item in _csv_set(notification_chat_ids)},
            notify_cycle_summary=notify_cycle_summary,
            notify_errors=notify_errors,
            daily_pnl_enabled=daily_pnl_enabled,
            pnl_report_interval_hours=max(0.01, float(pnl_report_interval_hours)),
            pnl_report_on_start=pnl_report_on_start,
        )


def canonical_command(raw: str) -> str:
    command = raw.strip().lower().replace("-", "_")
    return ALIASES.get(command, command)


def parse_command(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    first, _, rest = text.partition(" ")
    if first.startswith("/"):
        first = first[1:]
    first = first.split("@", 1)[0]
    return canonical_command(first), rest.strip()


def format_pnl_report(agent, *, interval_hours: float = 24.0, now: datetime.datetime | None = None) -> str:
    now = now or _utc_now()
    since = now - datetime.timedelta(hours=interval_hours)
    positions = agent.positions
    summary = positions.summary()
    risk = positions.risk_snapshot(now)
    realized_period = positions.realized_pnl_since(since)
    closed_period = positions.closed_count_since(since)
    opened_period = positions.opened_count_since(since)
    win_rate = summary["win_rate"]
    win_rate_text = "N/A" if win_rate is None else f"{win_rate:.1%}"
    return "\n".join(_table("ALBERT P&L REPORT", [
        ("Window", f"last {interval_hours:g}h"),
        ("Opened", opened_period),
        ("Closed", closed_period),
        ("Realized P&L", _signed_money(realized_period)),
        ("Closed P&L total", _signed_money(summary["total_pnl_usd"])),
        ("Unrealized P&L", _signed_money(summary["unrealized_pnl_usd"])),
        ("Net P&L", _signed_money(risk["net_pnl_usd"])),
        ("Open deployed", _money(summary["open_deployed_usd"])),
        ("Open planned risk", _money(summary["open_planned_risk_usd"])),
        ("Daily loss", _money(risk["daily_loss_usd"])),
        ("Drawdown", _money(risk["drawdown_usd"])),
        ("Win rate", win_rate_text),
    ]))


def format_cycle_report(agent, *, signals_count: int, duration_seconds: float, error: Exception | None = None) -> str:
    summary = agent.positions.summary()
    risk = agent.positions.risk_snapshot()
    if error is None:
        return "\n".join(_table("ALBERT CYCLE REPORT", [
            ("Status", "OK"),
            ("Errors", 0),
            ("Duration", f"{duration_seconds:.1f}s"),
            ("Signals", signals_count),
            ("Open positions", summary["open_positions"]),
            ("Deployed", _money(summary["open_deployed_usd"])),
            ("Unrealized P&L", _signed_money(summary["unrealized_pnl_usd"])),
            ("Net P&L", _signed_money(risk["net_pnl_usd"])),
            ("Daily loss", _money(risk["daily_loss_usd"])),
        ]))
    return "\n".join(_table("ALBERT CYCLE REPORT", [
        ("Status", "ERROR"),
        ("Errors", 1),
        ("Duration", f"{duration_seconds:.1f}s"),
        ("Error type", type(error).__name__),
        ("Message", str(error)[:500]),
    ]))


class RemoteControlCommandHandler:
    """Authorize and execute remote commands against an Albert agent."""

    def __init__(
        self,
        agent,
        policy: RemoteControlPolicy,
        *,
        days_ahead: int = 1,
        demo_runner: Optional[Callable[[], str]] = None,
    ) -> None:
        self.agent = agent
        self.policy = policy
        self.days_ahead = days_ahead
        self.demo_runner = demo_runner
        self.paused = False
        self._run_lock = threading.Lock()

    def handle(self, text: str, chat_id: int | str, username: str = "") -> str:
        command, _args = parse_command(text)
        if not command:
            return "Kirim /help untuk daftar command."

        if command in PUBLIC_COMMANDS:
            response = self._public_response(command, chat_id)
            self._audit(chat_id, username, command, True, "public")
            return response

        if not self.policy.is_chat_allowed(chat_id):
            self._audit(chat_id, username, command, False, "unauthorized_chat")
            return (
                "Akses ditolak. Chat ID ini belum diizinkan.\n"
                f"Chat ID: {chat_id}\n"
                "Tambahkan ID tersebut ke REMOTE_ALLOWED_CHAT_IDS melalui setup wizard lokal."
            )

        if not self.policy.is_command_allowed(command):
            self._audit(chat_id, username, command, False, "command_not_allowed")
            return f"Command '{command}' tidak diizinkan oleh REMOTE_ALLOWED_COMMANDS."

        try:
            response = self._execute(command)
            self._audit(chat_id, username, command, True, "ok")
            return _trim_message(response)
        except Exception as exc:
            logger.exception("Remote command failed: %s", command)
            self._audit(chat_id, username, command, True, f"error:{type(exc).__name__}")
            return f"Command '{command}' gagal: {exc}"

    def run_daemon_cycle(self) -> list:
        with self._run_lock:
            return self.agent.run_cycle(days_ahead=self.days_ahead)

    def _public_response(self, command: str, chat_id: int | str) -> str:
        if command == "whoami":
            return "\n".join(_table("ALBERT WHOAMI", [("Chat ID", chat_id)]))
        return self.help_text(chat_id)

    def help_text(self, chat_id: int | str | None = None) -> str:
        allowed = sorted(self.policy.allowed_commands)
        lines = [
            "ALBERT TELEGRAM CONTROL",
            "-----------------------",
            "Available safe commands:",
            "",
            "/status       portfolio and runtime status",
            "/positions    open/closed position summary",
            "/signals      latest cycle signals",
            "/learning     learning and persona stats",
            "/pnl          profit/loss report now",
            "/dry_run_once run one demo/paper cycle",
            "/demo_once    run one virtual demo cycle",
            "/pause        pause remote daemon loop",
            "/resume       resume remote daemon loop",
            "/whoami       show this Telegram chat ID",
            "",
            f"Allowlist: {', '.join(allowed) if allowed else '(empty)'}",
        ]
        if chat_id is not None and not self.policy.is_chat_allowed(chat_id):
            lines += [
                "",
                "This chat is not authorized yet.",
                "Run /whoami and add the ID to REMOTE_ALLOWED_CHAT_IDS.",
            ]
        return "\n".join(lines)

    def _execute(self, command: str) -> str:
        if command == "status":
            return self._status_text()
        if command == "positions":
            return self._positions_text()
        if command == "signals":
            return self._signals_text()
        if command == "learning":
            return self._learning_text()
        if command == "pnl":
            return self._pnl_text()
        if command == "pause":
            self.paused = True
            return "Albert remote daemon dipause. Command manual tetap bisa dipanggil dari allowlist."
        if command == "resume":
            self.paused = False
            return "Albert remote daemon dilanjutkan."
        if command == "dry_run_once":
            return self._dry_run_once()
        if command == "demo_once":
            return self._demo_once()
        return f"Command '{command}' belum diimplementasikan."

    def _status_text(self) -> str:
        state = getattr(self.agent, "_cycle_state", {})
        mode = "demo" if getattr(self.agent, "_demo", None) else ("dry" if self.agent.dry_run else "live")
        summary = self.agent.positions.summary()
        risk = self.agent.positions.risk_snapshot()
        runtime = _table("ALBERT STATUS", [
            ("Mode", mode),
            ("Paused", _yes_no(self.paused)),
            ("Cycle", state.get("cycle_num", 0)),
            ("Current city", state.get("current_city", "-")),
            ("Last signals", len(state.get("last_signals", []) or [])),
        ])
        portfolio = _table("PORTFOLIO", [
            ("Open positions", summary["open_positions"]),
            ("Open deployed", _money(summary["open_deployed_usd"])),
            ("Unrealized P&L", _signed_money(summary["unrealized_pnl_usd"])),
            ("Closed P&L", _signed_money(summary["total_pnl_usd"])),
            ("Net P&L", _signed_money(risk["net_pnl_usd"])),
            ("Daily loss", _money(risk["daily_loss_usd"])),
            ("Drawdown", _money(risk["drawdown_usd"])),
        ])
        return "\n".join(runtime + [""] + portfolio)

    def _positions_text(self) -> str:
        summary = self.agent.positions.summary()
        lines = _table("ALBERT POSITIONS", [
            ("Open", summary["open_positions"]),
            ("Closed", summary["closed_positions"]),
            ("Deployed", _money(summary["open_deployed_usd"])),
            ("Unrealized P&L", _signed_money(summary["unrealized_pnl_usd"])),
            ("Closed P&L", _signed_money(summary["total_pnl_usd"])),
        ])
        open_positions = list(self.agent.positions.open_positions.values())
        if not open_positions:
            lines += ["", "No open positions."]
            return "\n".join(lines)
        lines += ["", "OPEN POSITIONS", "--------------"]
        for idx, pos in enumerate(open_positions[:12], start=1):
            lines += [
                f"{idx}. {pos.city} {pos.direction} {pos.target_date}",
                f"   Bucket  : {_format_bucket(pos.bucket_low, pos.bucket_high)}",
                f"   Entry   : {pos.entry_price:.3f}",
                f"   Current : {pos.current_price:.3f}",
                f"   Size    : {_money(pos.size_usd)}",
                f"   PnL     : {pos.unrealized_pnl_pct:+.1%}",
                "",
            ]
        if len(open_positions) > 12:
            lines.append(f"... {len(open_positions) - 12} more positions")
        return "\n".join(lines)

    def _signals_text(self) -> str:
        signals = list(getattr(self.agent, "_cycle_state", {}).get("last_signals", []) or [])
        if not signals:
            return "ALBERT LAST SIGNALS\n-------------------\nNo signals from the latest cycle yet."
        lines = ["ALBERT LAST SIGNALS", "-------------------"]
        for idx, sig in enumerate(signals[:12], start=1):
            probability = getattr(sig, "model_probability", getattr(sig, "probability", 0.0))
            lines += [
                f"{idx}. {sig.city} {sig.direction} {sig.target_date}",
                f"   Bucket : {_format_bucket(sig.bucket_low, sig.bucket_high)}",
                f"   EV     : {sig.ev:+.3f}",
                f"   Prob   : {probability:.2f}",
                f"   Price  : {sig.market_price:.3f}",
                f"   Size   : {_money(sig.recommended_usd)}",
                "",
            ]
        if len(signals) > 12:
            lines.append(f"... {len(signals) - 12} more signals")
        return "\n".join(lines)

    def _learning_text(self) -> str:
        stats = self.agent.memory.overall_stats()
        total_pnl = float(stats.get("total_pnl_usd", 0) or 0)
        lines = _table("ALBERT LEARNING", [
            ("Predictions", stats.get("total_predictions", 0)),
            ("Avg Brier", stats.get("avg_brier_score", "N/A")),
            ("Trades", stats.get("total_trades", 0)),
            ("Win rate", stats.get("win_rate", "N/A")),
            ("Closed P&L", _signed_money(total_pnl)),
            ("Lessons", stats.get("total_lessons", 0)),
        ])
        scores = self.agent.memory.persona_score_report()
        if scores:
            lines += ["", "PERSONA WEIGHTS", "---------------"]
            for row in scores[:5]:
                lines.append(
                    f"{row['name']:<24} n={row['predictions']:<4} "
                    f"brier={row['avg_brier']} weight={row['weight']}"
                )
        return "\n".join(lines)

    def _pnl_text(self) -> str:
        return format_pnl_report(self.agent)

    def _dry_run_once(self) -> str:
        if self.demo_runner is not None:
            return self._demo_once()
        if self.paused:
            return "Albert sedang dipause. Jalankan /resume sebelum dry_run_once."
        if not self._run_lock.acquire(blocking=False):
            return "Albert sedang menjalankan cycle lain. Coba lagi setelah selesai."
        start = time.monotonic()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                signals = self.agent.run_cycle(days_ahead=self.days_ahead)
        finally:
            self._run_lock.release()
        return (
            "\n".join(_table("DRY-RUN COMPLETE", [
                ("Duration", f"{time.monotonic() - start:.1f}s"),
                ("Signals", len(signals)),
            ]))
            + "\n\n"
            f"{self._status_text()}"
        )

    def _demo_once(self) -> str:
        if self.paused:
            return "Albert sedang dipause. Jalankan /resume sebelum demo_once."
        if self.demo_runner is None:
            return "Demo runner belum dikonfigurasi."
        if not self._run_lock.acquire(blocking=False):
            return "Albert sedang menjalankan cycle lain. Coba lagi setelah selesai."
        start = time.monotonic()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.demo_runner()
        finally:
            self._run_lock.release()
        return (
            "\n".join(_table("DEMO COMPLETE", [
                ("Duration", f"{time.monotonic() - start:.1f}s"),
            ]))
            + "\n\n"
            + result
        )

    def _audit(self, chat_id: int | str, username: str, command: str, allowed: bool, outcome: str) -> None:
        if not self.policy.audit_log:
            return
        record = {
            "ts": _utc_now_iso(),
            "chat_id": str(chat_id),
            "username": username,
            "command": command,
            "allowed": allowed,
            "outcome": outcome,
        }
        try:
            parent = os.path.dirname(os.path.abspath(self.policy.audit_log))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.policy.audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("Could not write remote audit log: %s", exc)


class TelegramRemoteControlBot:
    """Long-polling Telegram adapter for RemoteControlCommandHandler."""

    def __init__(
        self,
        *,
        token: str,
        handler: RemoteControlCommandHandler,
        poll_interval_seconds: float = 2.0,
        request_timeout_seconds: int = 30,
        api_base: str = "https://api.telegram.org",
    ) -> None:
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required for Telegram remote control")
        self.token = token
        self.handler = handler
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.api_base = api_base.rstrip("/")
        self._offset: Optional[int] = None
        self._stopped = threading.Event()

    def stop(self) -> None:
        self._stopped.set()

    def run_forever(self) -> None:
        logger.info("Starting Telegram remote control polling")
        while not self._stopped.is_set():
            try:
                self.poll_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.warning("Telegram polling error: %s", exc)
                time.sleep(max(1.0, self.poll_interval_seconds))

    def poll_once(self) -> None:
        updates = self._get_updates()
        for update in updates:
            self._offset = int(update["update_id"]) + 1
            message = update.get("message") or update.get("edited_message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            text = message.get("text") or ""
            if chat_id is None or not text:
                continue
            user = message.get("from") or {}
            username = user.get("username") or user.get("first_name") or ""
            response = self.handler.handle(text, chat_id, username)
            self._send_message(chat_id, response)

    def send_notification(
        self,
        text: str,
        chat_ids: Iterable[int | str] | None = None,
    ) -> int:
        targets = [str(chat_id) for chat_id in (chat_ids or self.handler.policy.allowed_chat_ids)]
        sent = 0
        for chat_id in sorted(set(targets)):
            if not chat_id:
                continue
            self._send_message(chat_id, text)
            sent += 1
        return sent

    def _get_updates(self) -> list[dict]:
        params = {
            "timeout": int(self.request_timeout_seconds),
            "allowed_updates": json.dumps(["message", "edited_message"]),
        }
        if self._offset is not None:
            params["offset"] = self._offset
        resp = requests.get(
            self._url("getUpdates"),
            params=params,
            timeout=self.request_timeout_seconds + 5,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {payload}")
        return payload.get("result", [])

    def _send_message(self, chat_id: int | str, text: str) -> None:
        try:
            resp = requests.post(
                self._url("sendMessage"),
                json={
                    "chat_id": chat_id,
                    "text": _as_telegram_pre(text),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
        except Exception:
            logger.warning("Telegram sendMessage failed:\n%s", traceback.format_exc())

    def _url(self, method: str) -> str:
        return f"{self.api_base}/bot{self.token}/{method}"
