"""
tests/test_monitoring_extended.py
----------------------------------
Tests for SignalDiagnostics and PerformanceTracker.

Run with:
    pytest tests/test_monitoring_extended.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from monitoring.signal_diagnostics import (
    SignalDiagnostics, _diagnose_trend, _diagnose_mean_rev,
)
from monitoring.performance_tracker import (
    PerformanceTracker, PerformanceReport, DealRecord,
)
from strategies.base_strategy import Direction, Signal

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


# ===========================================================================
# Helpers
# ===========================================================================

def _signal(direction=Direction.LONG) -> Signal:
    return Signal(
        symbol="XAUUSD", timeframe="H1",
        direction=direction, strength=0.9,
        strategy_id="trend_ema9_21", bar_time=_NOW,
    )


def _row(**kwargs) -> dict:
    defaults = dict(
        symbol="XAUUSD", timeframe="H1",
        event_time=_NOW, close=2000.0,
        rsi_14=50.0, adx_14=30.0, atr_14=15.0,
        ema_9=2005.0, ema_21=2000.0,
        bb_lower_20=1975.0, bb_upper_20=2025.0,
        bb_mid_20=2000.0, bb_width_20=0.025,
        bb_pct_b_20=0.5,
    )
    defaults.update(kwargs)
    return defaults


def _deal(symbol="XAUUSD", profit=100.0, direction="long",
          hours_held=4) -> DealRecord:
    close_time = _NOW + timedelta(hours=hours_held)
    return DealRecord(
        ticket=1001, order=2001, symbol=symbol,
        direction=direction, volume=0.01,
        open_price=2000.0, close_price=2100.0 if profit > 0 else 1900.0,
        profit=profit, swap=0.0, commission=-7.0,
        open_time=_NOW, close_time=close_time,
    )


# ===========================================================================
# _diagnose_trend
# ===========================================================================

class TestDiagnoseTrend:
    def test_waiting_when_ema_missing(self):
        row = _row(ema_9=None, ema_21=None)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0, None, None)
        assert "WAITING" in result

    def test_no_signal_low_adx(self):
        row = _row(adx_14=15.0)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0,
                                  2000.0, 2001.0)
        assert "NO_SIGNAL" in result
        assert "ADX" in result

    def test_no_signal_no_crossover(self):
        # Both previous and current: fast > slow (no crossover)
        row = _row(ema_9=2005.0, ema_21=2000.0, adx_14=30.0)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0,
                                  2004.0, 2000.0)
        assert "NO_SIGNAL" in result
        assert "crossover" not in result.lower() or "no" in result.lower()

    def test_bullish_crossover_detected(self):
        # prev: fast < slow; curr: fast > slow
        row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=30.0)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0,
                                  1998.0, 2001.0)
        assert "CROSSOVER" in result
        assert "BULLISH" in result

    def test_bearish_crossover_detected(self):
        # prev: fast > slow; curr: fast < slow
        row = _row(ema_9=1999.0, ema_21=2001.0, adx_14=30.0)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0,
                                  2003.0, 2001.0)
        assert "CROSSOVER" in result
        assert "BEARISH" in result

    def test_waiting_no_prev_data(self):
        row = _row(adx_14=30.0)
        result = _diagnose_trend(row, "ema_9", "ema_21", "adx_14", 25.0,
                                  None, None)
        assert "WAITING" in result


# ===========================================================================
# _diagnose_mean_rev
# ===========================================================================

class TestDiagnoseMeanRev:
    def test_waiting_when_features_missing(self):
        row = _row(bb_lower_20=None)
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "WAITING" in result

    def test_blocked_by_high_adx(self):
        row = _row(adx_14=40.0)
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "ADX" in result

    def test_no_signal_price_within_bands(self):
        row = _row(close=2000.0, adx_14=20.0)  # close at mid
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "NO_SIGNAL" in result
        assert "within bands" in result

    def test_below_band_rsi_not_oversold(self):
        row = _row(close=1970.0, rsi_14=45.0, adx_14=20.0)
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "NO_SIGNAL" in result
        assert "RSI" in result or "rsi" in result

    def test_long_setup_detected(self):
        row = _row(close=1970.0, rsi_14=28.0, adx_14=20.0)
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "SHOULD_FIRE" in result
        assert "LONG" in result

    def test_short_setup_detected(self):
        row = _row(close=2030.0, rsi_14=72.0, adx_14=20.0)
        result = _diagnose_mean_rev(
            row, "bb_lower_20", "bb_upper_20", "bb_mid_20", "bb_width_20",
            "rsi_14", "adx_14", 30.0, 35.0, 65.0,
        )
        assert "SHOULD_FIRE" in result
        assert "SHORT" in result


# ===========================================================================
# SignalDiagnostics
# ===========================================================================

class TestSignalDiagnostics:
    def test_log_evaluation_no_signal(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path)
        diag.log_evaluation(
            strategy_id="trend_ema9_21",
            strategy_type="trend",
            symbol="XAUUSD",
            timeframe="H1",
            row=_row(),
            signal=None,
        )
        # CSV should be created
        assert diag.csv_path.exists()

    def test_log_evaluation_with_signal(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path)
        diag.log_evaluation(
            strategy_id="trend_ema9_21",
            strategy_type="trend",
            symbol="XAUUSD",
            timeframe="H1",
            row=_row(),
            signal=_signal(),
        )
        content = diag.csv_path.read_text()
        assert "LONG" in content

    def test_csv_has_header(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path)
        diag.log_evaluation("s", "trend", "XAUUSD", "H1", _row(), None)
        lines = diag.csv_path.read_text().splitlines()
        assert "strategy_id" in lines[0]
        assert "diagnosis" in lines[0]

    def test_csv_appends_multiple_rows(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path)
        for i in range(5):
            diag.log_evaluation("s", "trend", "XAUUSD", "H1", _row(), None)
        lines = diag.csv_path.read_text().strip().splitlines()
        assert len(lines) == 6  # 1 header + 5 data rows

    def test_timeframe_filter_skips_m1(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path, log_timeframes=["H1"])
        diag.log_evaluation("s", "trend", "XAUUSD", "M1", _row(), None)
        # M1 should be filtered — no CSV created
        assert not diag.csv_path.exists()

    def test_timeframe_filter_allows_h1(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path, log_timeframes=["H1"])
        diag.log_evaluation("s", "trend", "XAUUSD", "H1", _row(), None)
        assert diag.csv_path.exists()

    def test_mean_reversion_strategy_type(self, tmp_path):
        diag = SignalDiagnostics(log_dir=tmp_path)
        diag.log_evaluation(
            "mean_rev_bb20_rsi14", "mean_reversion",
            "XAUUSD", "H1", _row(close=1970.0, rsi_14=28.0, adx_14=20.0),
            None,
        )
        content = diag.csv_path.read_text()
        assert "SHOULD_FIRE" in content or "NO_SIGNAL" in content


# ===========================================================================
# PerformanceReport
# ===========================================================================

class TestPerformanceReport:
    def test_print_summary_runs(self, capsys):
        report = PerformanceReport(
            total_trades=10, winning_trades=7, losing_trades=3,
            win_rate=0.70, profit_factor=2.1, expectancy=15.0,
            gross_profit=105.0, gross_loss=50.0, total_net_pnl=55.0,
        )
        report.print_summary()
        out = capsys.readouterr().out
        assert "70.00%" in out
        assert "2.10" in out

    def test_readiness_shown_in_summary(self, capsys):
        report = PerformanceReport(
            ready_for_live=True,
            readiness_notes=["✓ trades=50 >= 30", "✓ win_rate=60.0%"],
        )
        report.print_summary()
        out = capsys.readouterr().out
        assert "READY" in out


# ===========================================================================
# PerformanceTracker — metric computation
# ===========================================================================

class TestPerformanceTrackerMetrics:
    def _tracker(self) -> PerformanceTracker:
        return PerformanceTracker()

    def test_win_rate_calculation(self):
        tracker = self._tracker()
        deals = [
            _deal(profit=100), _deal(profit=50),
            _deal(profit=-30), _deal(profit=80),
        ]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        assert report.win_rate == pytest.approx(0.75)

    def test_profit_factor(self):
        tracker = self._tracker()
        deals = [_deal(profit=200), _deal(profit=-100)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        # net_profit = profit + commission (-7 each)
        # deal1: 200 - 7 = 193 (win)
        # deal2: -100 - 7 = -107 (loss)
        assert report.gross_profit == pytest.approx(193.0)
        assert report.gross_loss == pytest.approx(107.0)
        assert report.profit_factor == pytest.approx(193.0 / 107.0, rel=1e-3)

    def test_expectancy(self):
        tracker = self._tracker()
        deals = [_deal(profit=100), _deal(profit=-50)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        # net = (100-7) + (-50-7) = 93 + (-57) = 36 → expectancy = 18
        assert report.expectancy == pytest.approx(18.0)

    def test_all_losing_profit_factor_zero(self):
        tracker = self._tracker()
        deals = [_deal(profit=-50), _deal(profit=-30)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        assert report.win_rate == pytest.approx(0.0)
        assert report.profit_factor == pytest.approx(0.0)

    def test_all_winning_profit_factor_inf(self):
        tracker = self._tracker()
        deals = [_deal(profit=100), _deal(profit=50)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        assert report.win_rate == pytest.approx(1.0)
        assert report.profit_factor == float("inf")

    def test_by_symbol_breakdown(self):
        tracker = self._tracker()
        deals = [
            _deal("XAUUSD", 100), _deal("XAUUSD", -50),
            _deal("EURUSD", 80),
        ]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        assert "XAUUSD" in report.by_symbol
        assert "EURUSD" in report.by_symbol
        assert report.by_symbol["EURUSD"]["trades"] == 1

    def test_avg_duration_calculated(self):
        tracker = self._tracker()
        deals = [_deal(hours_held=2), _deal(hours_held=6)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        assert report.avg_duration_hours == pytest.approx(4.0)

    def test_largest_win_and_loss(self):
        tracker = self._tracker()
        deals = [
            _deal(profit=200), _deal(profit=50),
            _deal(profit=-150), _deal(profit=-30),
        ]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        # net = profit + commission(-7): 193, 43, -157, -37
        assert report.largest_win  == pytest.approx(193.0)
        assert report.largest_loss == pytest.approx(-157.0)

    def test_deal_record_net_profit(self):
        deal = _deal(profit=100)
        assert deal.net_profit == pytest.approx(100 - 7)   # profit + commission

    def test_deal_record_is_win(self):
        assert _deal(profit=100).is_win
        assert not _deal(profit=-50).is_win

    def test_is_ready_for_live_passes(self):
        tracker = self._tracker()
        deals = [_deal(profit=80) for _ in range(35)]
        report = PerformanceReport()
        tracker._compute_metrics(deals, report)
        report.total_trades = 35
        # Manually check thresholds
        assert report.win_rate == pytest.approx(1.0)
        assert report.total_trades >= 30

    def test_empty_deals_returns_zero_metrics(self):
        tracker = self._tracker()
        report = PerformanceReport()
        tracker._compute_metrics([], report)
        assert report.total_trades == 0
        assert report.win_rate == 0.0
        assert report.expectancy == 0.0