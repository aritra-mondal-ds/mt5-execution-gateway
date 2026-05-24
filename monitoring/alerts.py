"""
monitoring/alerts.py
---------------------
Alert system for critical trading events.

Alert types:
  - DRAWDOWN_WARNING   : daily or peak drawdown approaching limit
  - DRAWDOWN_HALT      : circuit breaker triggered, trading stopped
  - TRADE_OPENED       : new position entered
  - TRADE_CLOSED       : position exited with P&L
  - SIGNAL_REJECTED    : risk engine rejected a signal
  - SYSTEM_ERROR       : unhandled exception in trading loop
  - CONNECTION_LOST    : broker or feed connection dropped

Delivery channels (configurable, all optional):
  - Console (loguru) — always active
  - Log file          — always active (via loguru)
  - Email             — requires SMTP config in .env
  - Desktop popup     — Windows toast notification (optional)

Design: alerts are fire-and-forget. A failed alert delivery
never crashes the trading loop — it is logged and swallowed.
"""

from __future__ import annotations

import smtplib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from enum import Enum
from typing import Callable

from loguru import logger


# ---------------------------------------------------------------------------
# Alert types and severity
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    DRAWDOWN_WARNING  = "drawdown_warning"
    DRAWDOWN_HALT     = "drawdown_halt"
    TRADE_OPENED      = "trade_opened"
    TRADE_CLOSED      = "trade_closed"
    SIGNAL_REJECTED   = "signal_rejected"
    SYSTEM_ERROR      = "system_error"
    CONNECTION_LOST   = "connection_lost"
    DAILY_SUMMARY     = "daily_summary"


@dataclass
class Alert:
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"[{ts}] [{self.level.value}] {self.alert_type.value}: {self.message}"


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Centralised alert dispatcher.

    Usage:
        alerts = AlertManager()
        alerts.configure_email(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            sender="bot@example.com",
            password="app_password",
            recipients=["you@example.com"],
        )

        # Fire alerts
        alerts.drawdown_warning(daily_dd=0.025, limit=0.03)
        alerts.trade_closed(symbol="XAUUSD", pnl=145.0, direction="long")
        alerts.system_error(error="Connection timeout", context="MT5Feed")
    """

    def __init__(
        self,
        min_level: AlertLevel = AlertLevel.WARNING,
        suppress_duplicate_seconds: int = 300,
    ) -> None:
        self._min_level = min_level
        self._suppress_seconds = suppress_duplicate_seconds
        self._email_config: dict | None = None
        self._custom_handlers: list[Callable[[Alert], None]] = []
        self._recent: dict[str, datetime] = {}   # alert_type → last sent time
        self._history: list[Alert] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipients: list[str],
        use_tls: bool = True,
    ) -> None:
        self._email_config = {
            "host": smtp_host,
            "port": smtp_port,
            "sender": sender,
            "password": password,
            "recipients": recipients,
            "use_tls": use_tls,
        }
        logger.info(f"AlertManager: email configured → {recipients}")

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a custom alert handler (e.g. Telegram, Slack)."""
        self._custom_handlers.append(handler)

    # ------------------------------------------------------------------
    # Convenience alert constructors
    # ------------------------------------------------------------------

    def drawdown_warning(
        self,
        daily_dd: float,
        peak_dd: float,
        daily_limit: float,
        peak_limit: float,
        equity: float,
    ) -> None:
        pct_of_daily = daily_dd / daily_limit if daily_limit > 0 else 0
        pct_of_peak  = peak_dd / peak_limit if peak_limit > 0 else 0
        worst = max(pct_of_daily, pct_of_peak)

        level = AlertLevel.CRITICAL if worst >= 1.0 else AlertLevel.WARNING
        atype = (AlertType.DRAWDOWN_HALT if worst >= 1.0
                 else AlertType.DRAWDOWN_WARNING)

        msg = (
            f"Daily DD: {daily_dd:.2%} / {daily_limit:.2%}  |  "
            f"Peak DD: {peak_dd:.2%} / {peak_limit:.2%}  |  "
            f"Equity: {equity:,.2f}"
        )
        self._fire(Alert(
            alert_type=atype, level=level, message=msg,
            data={"daily_dd": daily_dd, "peak_dd": peak_dd, "equity": equity},
        ))

    def trade_opened(
        self,
        symbol: str,
        direction: str,
        lots: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        strategy_id: str,
    ) -> None:
        msg = (
            f"{symbol} {direction.upper()} {lots:.2f} lots @ {entry_price:.4f}  |  "
            f"SL={stop_loss}  TP={take_profit}  |  [{strategy_id}]"
        )
        self._fire(Alert(
            alert_type=AlertType.TRADE_OPENED,
            level=AlertLevel.INFO, message=msg,
            data={"symbol": symbol, "direction": direction, "lots": lots,
                  "entry": entry_price, "sl": stop_loss, "tp": take_profit},
        ))

    def trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        lots: float,
        close_price: float,
        reason: str = "",
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} {symbol} {direction.upper()} CLOSED  |  "
            f"P&L: {pnl:+.2f}  |  lots={lots:.2f}  |  "
            f"exit={close_price:.4f}  |  {reason}"
        )
        level = AlertLevel.INFO
        self._fire(Alert(
            alert_type=AlertType.TRADE_CLOSED,
            level=level, message=msg,
            data={"symbol": symbol, "pnl": pnl, "reason": reason},
        ))

    def signal_rejected(
        self,
        strategy_id: str,
        symbol: str,
        reason: str,
    ) -> None:
        # Only log rejections at DEBUG level — very frequent
        logger.debug(
            f"Signal rejected | {strategy_id} | {symbol} | {reason}"
        )

    def system_error(self, error: str, context: str = "") -> None:
        msg = f"{context}: {error}" if context else error
        self._fire(Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.CRITICAL, message=msg,
            data={"error": error, "context": context},
        ))

    def connection_lost(self, source: str, error: str = "") -> None:
        msg = f"{source} connection lost. {error}"
        self._fire(Alert(
            alert_type=AlertType.CONNECTION_LOST,
            level=AlertLevel.CRITICAL, message=msg,
        ))

    def daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        n_trades: int,
        win_rate: float,
        daily_dd: float,
    ) -> None:
        direction = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{direction} Daily Summary  |  "
            f"Equity: {equity:,.2f}  |  "
            f"P&L: {daily_pnl:+.2f}  |  "
            f"Trades: {n_trades}  |  "
            f"Win rate: {win_rate:.0%}  |  "
            f"Max DD: {daily_dd:.2%}"
        )
        self._fire(Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            level=AlertLevel.INFO, message=msg,
            data={"equity": equity, "pnl": daily_pnl,
                  "trades": n_trades, "win_rate": win_rate},
        ))

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _fire(self, alert: Alert) -> None:
        """Dispatch an alert through all configured channels."""
        # Level filter
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL]
        if levels.index(alert.level) < levels.index(self._min_level):
            return

        # Duplicate suppression
        key = alert.alert_type.value
        now = datetime.now(tz=timezone.utc)
        last = self._recent.get(key)
        if last is not None:
            elapsed = (now - last).total_seconds()
            if elapsed < self._suppress_seconds:
                return
        self._recent[key] = now

        # Always log
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)
        log_fn(f"ALERT | {alert}")

        # Store history
        self._history.append(alert)
        if len(self._history) > 1000:
            self._history.pop(0)

        # Email
        if self._email_config:
            self._send_email(alert)

        # Custom handlers
        for handler in self._custom_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.warning(f"AlertManager: custom handler failed: {e}")

    def _send_email(self, alert: Alert) -> None:
        cfg = self._email_config
        if cfg is None:
            return
        try:
            body = str(alert)
            msg = MIMEText(body)
            msg["Subject"] = f"[TradingBot] {alert.level.value}: {alert.alert_type.value}"
            msg["From"] = cfg["sender"]
            msg["To"] = ", ".join(cfg["recipients"])

            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
                if cfg.get("use_tls"):
                    server.starttls()
                server.login(cfg["sender"], cfg["password"])
                server.sendmail(cfg["sender"], cfg["recipients"], msg.as_string())

            logger.debug(f"Alert email sent: {alert.alert_type.value}")

        except Exception as e:
            logger.warning(f"AlertManager: email failed: {e}")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    def recent_critical(self, n: int = 10) -> list[Alert]:
        return [
            a for a in reversed(self._history)
            if a.level == AlertLevel.CRITICAL
        ][:n]