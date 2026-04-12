"""
logger.py -- Structured, rotating trade and system event logger.

Writes JSON-structured log lines to a rotating file and optionally to stdout.
Provides domain-specific helpers (log_trade, log_fill, log_regime_change, etc.)
so that all critical events are consistently formatted and queryable.

Log record format (one JSON object per line):
  {"ts": "2024-01-15T10:32:01.123Z", "event": "trade", "level": "INFO", ...}
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# JSON formatter                                                               #
# --------------------------------------------------------------------------- #

class _JsonFormatter(logging.Formatter):
    """Emit each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts":      dt.datetime.utcfromtimestamp(record.created).strftime(
                           "%Y-%m-%dT%H:%M:%S.%f"
                       )[:-3] + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# --------------------------------------------------------------------------- #
# TradeLogger                                                                  #
# --------------------------------------------------------------------------- #

class TradeLogger:
    """
    Structured logger for regime-trader events.

    Creates a rotating file handler at ``log_dir/regime_trader.log`` and an
    optional console handler.  All log records are emitted as JSON lines.

    Parameters
    ----------
    log_dir      : Directory where log files are written.
    log_level    : Minimum log level: "DEBUG", "INFO", "WARNING", "ERROR".
    max_bytes    : Maximum log file size before rotation (bytes).
    backup_count : Number of rotated files to retain.
    console      : If True, also write records to stdout.
    """

    def __init__(
        self,
        log_dir:      str  = "logs/",
        log_level:    str  = "INFO",
        max_bytes:    int  = 50 * 1024 * 1024,
        backup_count: int  = 5,
        console:      bool = True,
    ) -> None:
        self.log_dir      = log_dir
        self.log_level    = log_level
        self.max_bytes    = max_bytes
        self.backup_count = backup_count
        self.console      = console

        self._logger: Optional[logging.Logger] = None

    # ======================================================================= #
    # Initialisation                                                           #
    # ======================================================================= #

    def setup(self) -> None:
        """Configure handlers and formatters.  Call once at startup."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log_path     = os.path.join(self.log_dir, "regime_trader.log")
        numeric_lvl  = getattr(logging, self.log_level.upper(), logging.INFO)

        self._logger = logging.getLogger("regime_trader")
        self._logger.setLevel(numeric_lvl)
        self._logger.propagate = False
        self._logger.handlers.clear()   # idempotent on re-setup

        # Rotating file handler (JSON lines)
        fh = logging.handlers.RotatingFileHandler(
            filename    = log_path,
            maxBytes    = self.max_bytes,
            backupCount = self.backup_count,
            encoding    = "utf-8",
        )
        fh.setLevel(numeric_lvl)
        fh.setFormatter(_JsonFormatter())
        self._logger.addHandler(fh)

        # Optional plain-text console handler
        if self.console:
            ch = logging.StreamHandler()
            ch.setLevel(numeric_lvl)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._logger.addHandler(ch)

        self._logger.info("TradeLogger ready  log=%s", log_path)

    # ======================================================================= #
    # Domain-specific helpers                                                  #
    # ======================================================================= #

    def log_trade(
        self,
        symbol:   str,
        side:     str,
        qty:      float,
        price:    float,
        order_id: Optional[str]          = None,
        extra:    Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order submission event."""
        self._emit(logging.INFO, "trade", {
            "symbol": symbol, "side": side,
            "qty": qty, "price": price,
            "order_id": order_id, **(extra or {}),
        })

    def log_fill(
        self,
        symbol:     str,
        side:       str,
        qty:        float,
        fill_price: float,
        order_id:   str,
        extra:      Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a confirmed order fill."""
        self._emit(logging.INFO, "fill", {
            "symbol": symbol, "side": side,
            "qty": qty, "fill_price": fill_price,
            "order_id": order_id, **(extra or {}),
        })

    def log_regime_change(
        self,
        previous_regime: str,
        new_regime:      str,
        confidence:      float,
        timestamp:       Optional[dt.datetime] = None,
        extra:           Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a regime transition event."""
        self._emit(logging.INFO, "regime_change", {
            "previous_regime": previous_regime,
            "new_regime":      new_regime,
            "confidence":      round(confidence, 4),
            "event_ts":        (timestamp or dt.datetime.utcnow()).isoformat(),
            **(extra or {}),
        })

    def log_rebalance(
        self,
        target_weights:   Dict[str, float],
        previous_weights: Dict[str, float],
        regime:           str,
        extra:            Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a portfolio rebalance event with before/after weights."""
        all_syms = set(target_weights) | set(previous_weights)
        deltas   = {
            s: round(target_weights.get(s, 0.0) - previous_weights.get(s, 0.0), 4)
            for s in all_syms
        }
        self._emit(logging.INFO, "rebalance", {
            "regime":           regime,
            "target_weights":   {k: round(v, 4) for k, v in target_weights.items()},
            "previous_weights": {k: round(v, 4) for k, v in previous_weights.items()},
            "deltas":           deltas,
            **(extra or {}),
        })

    def log_risk_event(
        self,
        event_type:  str,
        description: str,
        severity:    str = "WARNING",
        extra:       Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a risk-management event (drawdown breach, halt, size reduction)."""
        level = getattr(logging, severity.upper(), logging.WARNING)
        self._emit(level, "risk_event", {
            "event_type": event_type, "description": description,
            **(extra or {}),
        })

    def log_error(
        self,
        error:   Exception,
        context: Optional[str]           = None,
        extra:   Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an exception with optional context and stack trace."""
        self._emit(logging.ERROR, "error", {
            "error_type": type(error).__name__,
            "error_msg":  str(error),
            "context":    context,
            "traceback":  traceback.format_exc(),
            **(extra or {}),
        })

    # ======================================================================= #
    # Generic passthrough                                                      #
    # ======================================================================= #

    def info(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.INFO,    "info",    {"message": message, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, "warning", {"message": message, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR,   "error",   {"message": message, **kwargs})

    def debug(self, message: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG,   "debug",   {"message": message, **kwargs})

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _emit(self, level: int, event: str, payload: Dict[str, Any]) -> None:
        """
        Attach payload as extra_fields and emit via the underlying logger.
        Falls back to the root logger if setup() has not been called.
        """
        logger = self._logger or logging.getLogger("regime_trader")
        record = logger.makeRecord(
            name     = logger.name,
            level    = level,
            fn       = "",
            lno      = 0,
            msg      = f"[{event}] " + str(payload.get("message", "")),
            args     = (),
            exc_info = None,
        )
        record.extra_fields = {"event": event, **payload}  # type: ignore[attr-defined]
        logger.handle(record)
