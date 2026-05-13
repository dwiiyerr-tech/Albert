"""
TUI log handler — captures Python logging records into a rich-formatted
deque so the dashboard can display them in the center panel.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime

from rich.text import Text


class TuiLogHandler(logging.Handler):
    """
    Intercepts all log records and converts them to coloured rich Text
    objects stored in a fixed-size ring buffer.
    """

    _TAG_RULES: list[tuple[str, tuple[str, str]]] = [
        # (substring_to_match_in_lowercased_msg, (tag_label, rich_style))
        ("signal:",         ("TRADE", "bold blue")),
        ("place_order",     ("TRADE", "bold blue")),
        ("position opened", ("OK   ", "bold green")),
        ("position closed", ("OK   ", "bold green")),
        ("stop_loss",       ("STOP ", "bold red")),
        ("order",           ("API  ", "cyan")),
        ("filled",          ("API  ", "cyan")),
        ("lesson",          ("LEARN", "bold cyan")),
        ("pattern",         ("LEARN", "bold cyan")),
        ("calibrat",        ("LEARN", "bold cyan")),
        ("reflection",      ("LEARN", "bold cyan")),
        ("brier",           ("LEARN", "bold cyan")),
        ("debate",          ("SIM  ", "magenta")),
        ("persona",         ("SIM  ", "magenta")),
        ("simulation",      ("SIM  ", "magenta")),
        ("ecmwf",           ("TOOL ", "yellow")),
        ("gfs",             ("TOOL ", "yellow")),
        ("metar",           ("TOOL ", "yellow")),
        ("forecast",        ("TOOL ", "yellow")),
        ("fetch",           ("TOOL ", "yellow")),
        ("open_meteo",      ("TOOL ", "yellow")),
        ("═══",             ("CYCLE", "bold white")),
        ("processing",      ("CYCLE", "bold white")),
        ("cycle",           ("CYCLE", "bold white")),
    ]

    def __init__(self, maxlen: int = 300) -> None:
        super().__init__()
        self.buffer: deque[Text] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            msg_display = msg[:150]
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            msg_lower = msg.lower()

            if record.levelno >= logging.ERROR:
                tag, color = "ERROR", "bold red"
            elif record.levelno >= logging.WARNING:
                tag, color = "WARN ", "bold yellow"
            else:
                tag, color = "INFO ", "dim white"
                for fragment, (t, c) in self._TAG_RULES:
                    if fragment in msg_lower or fragment in msg:
                        tag, color = t, c
                        break

            entry = Text(overflow="ellipsis", no_wrap=True)
            entry.append(f"{time_str} ", style="dim")
            entry.append(f"{tag} ", style=color)
            entry.append(msg_display, style="white")
            self.buffer.append(entry)
        except Exception:
            pass
