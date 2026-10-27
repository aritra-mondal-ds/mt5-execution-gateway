"""
tests/test_scripts.py
----------------------
Tests for scripts/auto_optimizer.py and scripts/daily_report.py.

These tests mock MT5 so no live connection is required.

Run with:
    pytest tests/test_scripts.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _make_trades(n_wins: int, n_losses: int) -> list[dict]:
    trades = []
    for i in range(n_wins):
        trades.append({
            "symbol": "XAUUSD", "direction": "long",
            "net_pnl": 50.0 + i,
            "open_time": _NOW + timedelta(hours=i),
            "close_time": _NOW + timedelta(hours=i, minutes=30),
        })
    for i in range(n_losses):
        trades.append({
            "symbol": "XAUUSD", "direction": "short",
            "net_pnl": -30.0 - i,
            "open_time": _NOW + timedelta(hours=n_wins + i),
            "close_time": _NOW + timedelta(hours=n_wins + i, minutes=20),
        })
    return trades


# ===========================================================================
# auto_optimizer
# ===========================================================================

class TestAutoOptimizer:
    def test_load_params(self, tmp_path):
        from scripts.auto_optimizer import load_params, save_params, PARAMS_FILE
        import scripts.auto_optimizer as ao

        params_file = tmp_path / "live_params.json"
        params = {
            "_version": 1, "_last_updated": "test", "_updated_by": "test",
            "trend": {"adx_threshold": 20.0, "atr_multiplier": 1.5, "tp_multiplier": 3.0},
            "mean_reversion": {"oversold_rsi": 35.0, "overbought_rsi": 65.0,
                               "adx_ceiling": 25.0, "max_bars_in_trade": 24},
            "risk": {},
        }
        params_file.write_text(json.dumps(params))

        # Monkey-patch PARAMS_FILE
        original = ao.PARAMS_FILE
        ao.PARAMS_FILE = params_file
        try:
            loaded = load_params()
            assert loaded["trend"]["adx_threshold"] == pytest.approx(20.0)
        finally:
            ao.PARAMS_FILE = original

    def test_save_params_increments_version(self, tmp_path):
        import scripts.auto_optimizer as ao
        params_file = tmp_path / "live_params.json"
        params = {"_version": 2, "trend": {}, "mean_reversion": {}, "risk": {}}
        params_file.write_text(json.dumps(params))
        original = ao.PARAMS_FILE
        ao.PARAMS_FILE = params_file
        try:
            from scripts.auto_optimizer import save_params
            save_params(params, "test reason")
            saved = json.loads(params_file.read_text())
            assert saved["_version"] == 3
            assert "test reason" in saved["_updated_by"]
        finally:
            ao.PARAMS_FILE = original

    def test_expectancy_calculation(self):
        from scripts.auto_optimizer import _expectancy
        trades = _make_trades(3, 1)
        exp = _expectancy(trades)
        # wins: 50+51+52=153, loss: -30 → avg = 123/4 = 30.75
        assert exp == pytest.approx(30.75)

    def test_win_rate_calculation(self):
        from scripts.auto_optimizer import _win_rate
        trades = _make_trades(7, 3)
        assert _win_rate(trades) == pytest.approx(0.70)

    def test_empty_trades_expectancy(self):
        from scripts.auto_optimizer import _expectancy
        assert _expectancy([]) == pytest.approx(-999.0)

    def test_empty_trades_win_rate(self):
        from scripts.auto_optimizer import _win_rate
        assert _win_rate([]) == pytest.approx(0.0)

    def test_objective_trend_scores_positive_for_wins(self):
        import optuna
        from scripts.auto_optimizer import objective_trend
        trades = _make_trades(8, 2)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: objective_trend(t, trades),
            n_trials=5, show_progress_bar=False,
        )
        assert study.best_value > -999.0

    def test_objective_mean_rev_scores(self):
        import optuna
        from scripts.auto_optimizer import objective_mean_rev
        trades = _make_trades(6, 4)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: objective_mean_rev(t, trades),
            n_trials=5, show_progress_bar=False,
        )
        assert study.best_value > -999.0

    def test_run_optimization_skips_with_no_trades(self, tmp_path):
        import scripts.auto_optimizer as ao
        original = ao.PARAMS_FILE

        params_file = tmp_path / "live_params.json"
        params = {
            "_version": 1, "_last_updated": "", "_updated_by": "",
            "trend": {"adx_threshold": 20.0, "atr_multiplier": 1.5, "tp_multiplier": 3.0},
            "mean_reversion": {"oversold_rsi": 35.0, "overbought_rsi": 65.0,
                               "adx_ceiling": 25.0, "max_bars_in_trade": 24},
            "risk": {},
        }
        params_file.write_text(json.dumps(params))
        ao.PARAMS_FILE = params_file

        try:
            with patch.object(ao, "get_closed_trades", return_value=[]):
                updated = ao.run_optimization(force=False)
            assert not updated
        finally:
            ao.PARAMS_FILE = original

    def test_run_optimization_with_enough_trades(self, tmp_path):
        import scripts.auto_optimizer as ao
        original = ao.PARAMS_FILE
        params_file = tmp_path / "live_params.json"
        params = {
            "_version": 1, "_last_updated": "", "_updated_by": "",
            "trend": {"adx_threshold": 20.0, "atr_multiplier": 1.5, "tp_multiplier": 3.0,
                      "fast_period": 9, "slow_period": 21, "exit_adx_threshold": 15.0,
                      "cooldown_bars": 2},
            "mean_reversion": {"oversold_rsi": 35.0, "overbought_rsi": 65.0,
                               "adx_ceiling": 25.0, "max_bars_in_trade": 24,
                               "atr_multiplier": 2.0, "min_band_width": 0.001},
            "risk": {},
        }
        params_file.write_text(json.dumps(params))
        ao.PARAMS_FILE = params_file
        ao.N_TRIALS = 3  # fast for testing

        trades = _make_trades(15, 5)
        try:
            with patch.object(ao, "get_closed_trades", return_value=trades):
                ao.run_optimization(force=True)
            # File should still be valid JSON regardless of whether updated
            result = json.loads(params_file.read_text())
            assert "trend" in result
        finally:
            ao.PARAMS_FILE = original
            ao.N_TRIALS = 30


# ===========================================================================
# daily_report
# ===========================================================================

class TestDailyReport:
    def test_build_report_no_mt5(self):
        """Report builds without crashing when MT5 is unavailable."""
        from scripts.daily_report import build_report
        with patch("scripts.daily_report._get_account_snapshot", return_value={}), \
             patch("scripts.daily_report._get_todays_trades", return_value=[]), \
             patch("scripts.daily_report._get_all_time_trades", return_value=[]), \
             patch("scripts.daily_report._read_params", return_value={}), \
             patch("scripts.daily_report._read_signal_diagnostics_today", return_value={}):
            report = build_report()
        assert "DAILY TRADING REPORT" in report
        assert "ACCOUNT" in report

    def test_report_with_trades(self):
        from scripts.daily_report import build_report
        mock_trades = [
            {"symbol": "XAUUSD", "direction": "LONG", "lots": 1.0,
             "open": 3200.0, "close": 3210.0, "net_pnl": 100.0,
             "hold_h": 2.5, "win": True},
            {"symbol": "EURUSD", "direction": "SHORT", "lots": 0.5,
             "open": 1.085, "close": 1.088, "net_pnl": -30.0,
             "hold_h": 1.0, "win": False},
        ]
        with patch("scripts.daily_report._get_account_snapshot",
                   return_value={"equity": 100070.0, "balance": 100000.0,
                                 "profit": 70.0, "margin": 0.0,
                                 "n_open": 0, "open_syms": []}), \
             patch("scripts.daily_report._get_todays_trades", return_value=mock_trades), \
             patch("scripts.daily_report._get_all_time_trades", return_value=mock_trades), \
             patch("scripts.daily_report._read_params", return_value={}), \
             patch("scripts.daily_report._read_signal_diagnostics_today", return_value={}):
            report = build_report()

        assert "100,070.00" in report
        assert "50%" in report   # win rate
        assert "XAUUSD" in report
        assert "+100.00" in report or "100.00" in report

    def test_report_recommendation_not_enough_trades(self):
        from scripts.daily_report import build_report
        with patch("scripts.daily_report._get_account_snapshot", return_value={}), \
             patch("scripts.daily_report._get_todays_trades", return_value=[]), \
             patch("scripts.daily_report._get_all_time_trades", return_value=[]), \
             patch("scripts.daily_report._read_params", return_value={}), \
             patch("scripts.daily_report._read_signal_diagnostics_today", return_value={}):
            report = build_report()
        assert "Accumulating" in report or "10+" in report

    def test_report_recommendation_performing_well(self):
        from scripts.daily_report import build_report
        good_trades = [{"net_pnl": 80.0, "symbol": "XAUUSD"} for _ in range(20)]
        with patch("scripts.daily_report._get_account_snapshot", return_value={}), \
             patch("scripts.daily_report._get_todays_trades", return_value=[]), \
             patch("scripts.daily_report._get_all_time_trades", return_value=good_trades), \
             patch("scripts.daily_report._read_params", return_value={}), \
             patch("scripts.daily_report._read_signal_diagnostics_today", return_value={}):
            report = build_report()
        assert "performing well" in report or "No changes" in report

    def test_save_report(self, tmp_path):
        from scripts import daily_report as dr
        original_dir = dr.REPORT_DIR
        dr.REPORT_DIR = tmp_path
        try:
            path = dr.save_report("Test report content")
            assert path.exists()
            assert "Test report content" in path.read_text()
        finally:
            dr.REPORT_DIR = original_dir