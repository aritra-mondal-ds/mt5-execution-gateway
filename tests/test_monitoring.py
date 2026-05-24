"""
tests/test_monitoring.py
-------------------------
Tests for the monitoring layer.

Coverage:
  - AlertManager: fire, level filter, duplicate suppression,
    custom handlers, history, email (mocked)

The Streamlit dashboard (dashboard.py) is not unit-tested here —
it is validated by running it manually:
    streamlit run monitoring/dashboard.py

Run with:
    pytest tests/test_monitoring.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from monitoring.alerts import (
    AlertManager, Alert, AlertLevel, AlertType,
)


# ===========================================================================
# Alert dataclass
# ===========================================================================

class TestAlert:
    def test_str_contains_type_and_message(self):
        alert = Alert(
            alert_type=AlertType.TRADE_CLOSED,
            level=AlertLevel.INFO,
            message="XAUUSD closed +145.00",
        )
        s = str(alert)
        assert "trade_closed" in s
        assert "XAUUSD closed" in s
        assert "INFO" in s

    def test_timestamp_is_utc_aware(self):
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.CRITICAL,
            message="test",
        )
        assert alert.timestamp.tzinfo is not None

    def test_data_field_defaults_to_empty_dict(self):
        alert = Alert(
            alert_type=AlertType.DRAWDOWN_WARNING,
            level=AlertLevel.WARNING,
            message="test",
        )
        assert alert.data == {}


# ===========================================================================
# AlertManager — core dispatch
# ===========================================================================

class TestAlertManagerDispatch:
    def _manager(self, min_level=AlertLevel.INFO, suppress=0) -> AlertManager:
        return AlertManager(
            min_level=min_level,
            suppress_duplicate_seconds=suppress,
        )

    def test_alert_added_to_history(self):
        am = self._manager()
        am.system_error("test error")
        assert len(am.history) == 1
        assert am.history[0].alert_type == AlertType.SYSTEM_ERROR

    def test_level_filter_blocks_below_min(self):
        am = self._manager(min_level=AlertLevel.WARNING)
        # INFO-level alert (trade_opened) should be blocked
        am.trade_opened("XAUUSD", "long", 0.1, 2000.0, 1985.0, 2030.0, "strat")
        assert len(am.history) == 0

    def test_level_filter_passes_at_min(self):
        am = self._manager(min_level=AlertLevel.WARNING)
        am.system_error("crash")   # CRITICAL >= WARNING → passes
        assert len(am.history) == 1

    def test_critical_always_passes_warning_filter(self):
        am = self._manager(min_level=AlertLevel.WARNING)
        am.drawdown_warning(
            daily_dd=0.04, peak_dd=0.09,
            daily_limit=0.03, peak_limit=0.08,
            equity=9_500.0,
        )
        assert len(am.history) == 1
        assert am.history[0].level == AlertLevel.CRITICAL

    def test_duplicate_suppression(self):
        am = self._manager(suppress=300)
        am.system_error("error 1")
        am.system_error("error 2")   # same type, within suppress window
        assert len(am.history) == 1  # second suppressed

    def test_no_suppression_when_zero(self):
        am = self._manager(suppress=0)
        am.system_error("error 1")
        am.system_error("error 2")
        assert len(am.history) == 2

    def test_custom_handler_called(self):
        received = []
        am = self._manager()
        am.add_handler(lambda a: received.append(a))
        am.system_error("crash")
        assert len(received) == 1
        assert received[0].alert_type == AlertType.SYSTEM_ERROR

    def test_custom_handler_exception_does_not_crash(self):
        """A broken handler must not stop other handlers or crash the system."""
        am = self._manager()
        am.add_handler(lambda a: (_ for _ in ()).throw(RuntimeError("handler broke")))
        # Should not raise
        am.system_error("test")
        assert len(am.history) == 1

    def test_multiple_custom_handlers(self):
        received_a, received_b = [], []
        am = self._manager()
        am.add_handler(lambda a: received_a.append(a))
        am.add_handler(lambda a: received_b.append(a))
        am.system_error("test")
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_history_capped_at_1000(self):
        am = self._manager(suppress=0)
        # Fire 1100 different alert types by resetting _recent
        for i in range(1100):
            am._recent.clear()  # bypass suppression
            am._fire(Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.CRITICAL,
                message=f"error {i}",
            ))
        assert len(am.history) == 1000


# ===========================================================================
# AlertManager — convenience methods
# ===========================================================================

class TestAlertManagerMethods:
    def _manager(self) -> AlertManager:
        return AlertManager(min_level=AlertLevel.INFO, suppress_duplicate_seconds=0)

    def test_trade_opened_fires_info(self):
        am = self._manager()
        am.trade_opened("XAUUSD", "long", 0.1, 2000.0, 1985.0, 2030.0, "trend")
        assert len(am.history) == 1
        assert am.history[0].alert_type == AlertType.TRADE_OPENED
        assert am.history[0].level == AlertLevel.INFO

    def test_trade_closed_fires_info(self):
        am = self._manager()
        am.trade_closed("XAUUSD", "long", pnl=145.0, lots=0.1,
                        close_price=2014.5, reason="tp")
        assert len(am.history) == 1
        assert "145" in am.history[0].message

    def test_trade_closed_message_has_pnl_sign(self):
        am = self._manager()
        am.trade_closed("XAUUSD", "long", pnl=-50.0, lots=0.1,
                        close_price=1995.0, reason="sl")
        msg = am.history[0].message
        assert "❌" in msg or "-50" in msg

    def test_drawdown_warning_level(self):
        am = self._manager()
        # Under limit → WARNING
        am.drawdown_warning(
            daily_dd=0.02, peak_dd=0.04,
            daily_limit=0.03, peak_limit=0.08,
            equity=9_800.0,
        )
        assert am.history[0].level == AlertLevel.WARNING
        assert am.history[0].alert_type == AlertType.DRAWDOWN_WARNING

    def test_drawdown_halt_level(self):
        am = self._manager()
        # Over daily limit → CRITICAL + HALT type
        am.drawdown_warning(
            daily_dd=0.035, peak_dd=0.04,
            daily_limit=0.03, peak_limit=0.08,
            equity=9_650.0,
        )
        assert am.history[0].level == AlertLevel.CRITICAL
        assert am.history[0].alert_type == AlertType.DRAWDOWN_HALT

    def test_connection_lost_is_critical(self):
        am = self._manager()
        am.connection_lost("MT5Feed", "Timeout after 10s")
        assert am.history[0].level == AlertLevel.CRITICAL
        assert "MT5Feed" in am.history[0].message

    def test_daily_summary_fires(self):
        am = self._manager()
        am.daily_summary(
            equity=10_500.0, daily_pnl=500.0,
            n_trades=5, win_rate=0.6, daily_dd=0.01,
        )
        assert len(am.history) == 1
        assert am.history[0].alert_type == AlertType.DAILY_SUMMARY
        assert "10,500" in am.history[0].message

    def test_signal_rejected_does_not_add_to_history(self):
        """signal_rejected is debug-only, not stored in history."""
        am = self._manager()
        am.signal_rejected("trend_ema9_21", "XAUUSD", "drawdown_halt")
        assert len(am.history) == 0

    def test_recent_critical_returns_only_critical(self):
        am = self._manager()
        am.trade_opened("XAUUSD", "long", 0.1, 2000.0, None, None, "s")
        am._recent.clear()
        am.system_error("crash 1")
        am._recent.clear()
        am.system_error("crash 2")
        criticals = am.recent_critical()
        assert all(a.level == AlertLevel.CRITICAL for a in criticals)
        assert len(criticals) == 2


# ===========================================================================
# AlertManager — email (mocked)
# ===========================================================================

class TestAlertManagerEmail:
    def test_email_configured(self):
        am = AlertManager()
        am.configure_email(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            sender="bot@test.com",
            password="secret",
            recipients=["trader@test.com"],
        )
        assert am._email_config is not None
        assert am._email_config["recipients"] == ["trader@test.com"]

    def test_email_sent_on_critical(self):
        am = AlertManager(min_level=AlertLevel.INFO)
        am.configure_email(
            smtp_host="smtp.test.com", smtp_port=587,
            sender="bot@test.com", password="pw",
            recipients=["you@test.com"],
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            am.system_error("critical failure")

            # SMTP was constructed
            mock_smtp.assert_called_once()

    def test_email_failure_does_not_crash(self):
        """SMTP errors must be swallowed — never crash the trading loop."""
        am = AlertManager(min_level=AlertLevel.INFO)
        am.configure_email(
            smtp_host="bad.host", smtp_port=587,
            sender="bot@test.com", password="pw",
            recipients=["you@test.com"],
        )

        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            # Must not raise
            am.system_error("test error")

        assert len(am.history) == 1  # alert still stored despite email failure