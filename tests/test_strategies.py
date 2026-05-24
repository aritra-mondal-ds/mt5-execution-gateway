"""
tests/test_strategies.py
-------------------------
Tests for the complete strategy layer.

Coverage:
  - Signal dataclass invariants
  - BaseStrategy lifecycle (warmup, on_bar_update, on_fill, reset)
  - TrendStrategy: entry/exit logic, cooldown, ADX filter
  - MeanReversionStrategy: band/RSI entry, mean reversion exit, max hold
  - MLStrategy: inference routing, threshold logic, missing-feature safety

All tests use synthetic feature rows — no live data required.

Run with:
    pytest tests/test_strategies.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from strategies.base_strategy import BaseStrategy, Direction, Signal, StrategyState
from strategies.trend import TrendStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy


# ===========================================================================
# Shared helpers
# ===========================================================================

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _row(**kwargs) -> dict:
    """
    Build a minimal feature row with sensible XAUUSD defaults.
    Override any field via kwargs.
    """
    defaults = dict(
        symbol="XAUUSD",
        timeframe="H1",
        event_time=_NOW,
        open=2000.0,
        high=2010.0,
        low=1990.0,
        close=2005.0,
        volume=10000.0,
        spread=2.0,
        # Technical indicators
        ema_9=2005.0,
        ema_21=2000.0,
        ema_50=1990.0,
        ema_200=1970.0,
        rsi_14=50.0,
        atr_14=15.0,
        adx_14=30.0,
        pdi_14=25.0,
        ndi_14=20.0,
        bb_lower_20=1975.0,
        bb_upper_20=2025.0,
        bb_mid_20=2000.0,
        bb_width_20=0.025,
        macd_line=2.0,
        macd_signal=1.5,
        macd_hist=0.5,
    )
    defaults.update(kwargs)
    return defaults


def _advance_time(row: dict, hours: int = 1) -> dict:
    """Return a copy of row with event_time advanced by `hours`."""
    r = dict(row)
    r["event_time"] = row["event_time"] + timedelta(hours=hours)
    return r


# ===========================================================================
# Signal tests
# ===========================================================================

class TestSignal:
    def test_basic_construction(self):
        s = Signal(
            symbol="XAUUSD", timeframe="H1",
            direction=Direction.LONG, strength=0.8,
            strategy_id="test", bar_time=_NOW,
        )
        assert s.is_entry
        assert not s.is_exit
        assert s.direction == Direction.LONG

    def test_flat_is_exit(self):
        s = Signal(
            symbol="XAUUSD", timeframe="H1",
            direction=Direction.FLAT, strength=1.0,
            strategy_id="test", bar_time=_NOW,
        )
        assert s.is_exit
        assert not s.is_entry

    def test_strength_bounds(self):
        with pytest.raises(AssertionError):
            Signal(
                symbol="XAUUSD", timeframe="H1",
                direction=Direction.LONG, strength=1.5,  # > 1.0
                strategy_id="test", bar_time=_NOW,
            )
        with pytest.raises(AssertionError):
            Signal(
                symbol="XAUUSD", timeframe="H1",
                direction=Direction.LONG, strength=-0.1,  # < 0.0
                strategy_id="test", bar_time=_NOW,
            )

    def test_bar_time_must_be_utc_aware(self):
        with pytest.raises(AssertionError):
            Signal(
                symbol="XAUUSD", timeframe="H1",
                direction=Direction.LONG, strength=0.8,
                strategy_id="test",
                bar_time=datetime(2024, 1, 1),  # naive — should fail
            )

    def test_repr_contains_key_info(self):
        s = Signal(
            symbol="XAUUSD", timeframe="H1",
            direction=Direction.LONG, strength=0.75,
            strategy_id="my_strat", bar_time=_NOW,
        )
        r = repr(s)
        assert "my_strat" in r
        assert "LONG" in r
        assert "XAUUSD" in r


# ===========================================================================
# BaseStrategy lifecycle
# ===========================================================================

class ConcreteStrategy(BaseStrategy):
    """Minimal implementation for testing the base class."""

    def __init__(self, **kwargs):
        super().__init__(
            strategy_id="test_strat",
            symbol="XAUUSD",
            timeframe="H1",
            warmup_bars=kwargs.get("warmup_bars", 5),
        )
        self.signals_emitted = []

    def on_bar(self, row: dict) -> Signal | None:
        # Always emit a LONG signal for testing
        sig = self._make_signal(row, Direction.LONG)
        self.signals_emitted.append(sig)
        return sig

    def reset(self) -> None:
        from strategies.base_strategy import StrategyState
        self._state = StrategyState(strategy_id=self.strategy_id)
        self._bar_count = 0
        self.signals_emitted = []


class TestBaseStrategy:
    def test_warmup_suppresses_signals(self):
        strat = ConcreteStrategy(warmup_bars=5)
        row = _row()
        # First 4 bars: should return None (< warmup_bars)
        for _ in range(4):
            assert strat.on_bar_update(row) is None
        # 5th bar: warmup complete, signal emitted
        assert strat.on_bar_update(row) is not None

    def test_bar_count_increments(self):
        strat = ConcreteStrategy(warmup_bars=2)
        strat.on_bar_update(_row())
        strat.on_bar_update(_row())
        assert strat._bar_count == 2

    def test_on_fill_updates_position(self):
        strat = ConcreteStrategy(warmup_bars=1)
        row = _row()
        sig = strat.on_bar_update(row)
        assert sig is not None
        strat.on_fill(sig, fill_price=2005.0)
        assert strat.state.current_position == Direction.LONG

    def test_reset_clears_state(self):
        strat = ConcreteStrategy(warmup_bars=1)
        strat.on_bar_update(_row())
        strat.reset()
        assert strat._bar_count == 0
        assert strat.state.current_position == Direction.FLAT
        assert len(strat.signals_emitted) == 0

    def test_make_signal_fills_atr(self):
        strat = ConcreteStrategy(warmup_bars=1)
        row = _row(atr_14=20.0)
        strat.on_bar_update(row)
        assert strat.signals_emitted[-1].atr_at_signal == 20.0

    def test_get_returns_default_on_none(self):
        strat = ConcreteStrategy()
        row = _row()
        row["missing_col"] = None
        assert strat._get(row, "missing_col", default=99.0) == 99.0
        assert strat._get(row, "totally_absent", default=-1) == -1


# ===========================================================================
# TrendStrategy
# ===========================================================================

class TestTrendStrategy:
    def _make_strat(self, **kwargs) -> TrendStrategy:
        defaults = dict(
            symbol="XAUUSD", timeframe="H1",
            fast_period=9, slow_period=21,
            adx_period=14, cooldown_bars=0,
        )
        defaults.update(kwargs)   # kwargs wins — no duplicate-key error
        return TrendStrategy(**defaults)

    def _bypass_warmup(self, strat: TrendStrategy, row: dict) -> None:
        """Feed warmup_bars rows to satisfy the warmup guard."""
        strat._bar_count = strat.warmup_bars  # shortcut

    def _set_prev_emas(self, strat: TrendStrategy, fast: float, slow: float) -> None:
        """Directly set previous EMA state for crossover testing."""
        strat._prev_fast = fast
        strat._prev_slow = slow

    def test_no_signal_before_warmup(self):
        strat = self._make_strat()
        assert strat.on_bar_update(_row()) is None

    def test_long_entry_on_bullish_crossover(self):
        strat = self._make_strat()
        self._bypass_warmup(strat, _row())
        # Previous: fast < slow → bearish
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        # Current: fast > slow → bullish crossover
        row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=30.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_short_entry_on_bearish_crossover(self):
        strat = self._make_strat()
        self._bypass_warmup(strat, _row())
        # Previous: fast > slow → bullish
        self._set_prev_emas(strat, fast=2005.0, slow=2001.0)
        # Current: fast < slow → bearish crossover
        row = _row(ema_9=1999.0, ema_21=2001.0, adx_14=30.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.SHORT

    def test_no_entry_when_adx_below_threshold(self):
        strat = self._make_strat(adx_threshold=25.0)
        self._bypass_warmup(strat, _row())
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        # Crossover present but ADX too low (choppy)
        row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=15.0)
        sig = strat.on_bar(row)
        assert sig is None

    def test_stop_loss_set_below_entry_for_long(self):
        strat = self._make_strat(atr_multiplier=1.5)
        self._bypass_warmup(strat, _row())
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=30.0, atr_14=20.0, close=2005.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.stop_loss == pytest.approx(2005.0 - 1.5 * 20.0)

    def test_take_profit_set_above_entry_for_long(self):
        strat = self._make_strat(atr_multiplier=1.5, tp_multiplier=3.0)
        self._bypass_warmup(strat, _row())
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=30.0, atr_14=20.0, close=2005.0)
        sig = strat.on_bar(row)
        assert sig.take_profit == pytest.approx(2005.0 + 3.0 * 20.0)

    def test_exit_on_reverse_crossover(self):
        strat = self._make_strat()
        self._bypass_warmup(strat, _row())
        # Enter long
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        long_row = _row(ema_9=2003.0, ema_21=2001.0, adx_14=30.0)
        sig = strat.on_bar(long_row)
        assert sig is not None and sig.direction == Direction.LONG
        strat._state.current_position = Direction.LONG

        # Now bearish crossover → should exit
        strat._prev_fast = 2003.0
        strat._prev_slow = 2001.0
        exit_row = _row(ema_9=1999.0, ema_21=2001.0, adx_14=30.0)
        sig = strat.on_bar(exit_row)
        assert sig is not None
        assert sig.direction == Direction.FLAT

    def test_exit_on_adx_exhaustion(self):
        strat = self._make_strat(exit_adx_threshold=20.0)
        self._bypass_warmup(strat, _row())
        strat._state.current_position = Direction.LONG
        strat._prev_fast = 2003.0
        strat._prev_slow = 2001.0
        # Same crossover direction but ADX dropped
        row = _row(ema_9=2004.0, ema_21=2001.0, adx_14=10.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.meta.get("exit_reason") == "adx_exhaustion"

    def test_no_signal_when_features_missing(self):
        strat = self._make_strat()
        self._bypass_warmup(strat, _row())
        self._set_prev_emas(strat, fast=1998.0, slow=2002.0)
        row = _row(ema_9=None, ema_21=2001.0, adx_14=30.0)
        assert strat.on_bar(row) is None

    def test_reset_clears_ema_state(self):
        strat = self._make_strat()
        strat._prev_fast = 2000.0
        strat._prev_slow = 1999.0
        strat.reset()
        assert strat._prev_fast is None
        assert strat._prev_slow is None


# ===========================================================================
# MeanReversionStrategy
# ===========================================================================

class TestMeanReversionStrategy:
    def _make_strat(self, **kwargs) -> MeanReversionStrategy:
        defaults = dict(
            symbol="XAUUSD", timeframe="H1",
            bb_period=20, rsi_period=14,
        )
        defaults.update(kwargs)   # kwargs wins — no duplicate-key error
        return MeanReversionStrategy(**defaults)

    def _bypass_warmup(self, strat: MeanReversionStrategy) -> None:
        strat._bar_count = strat.warmup_bars

    def test_no_signal_before_warmup(self):
        strat = self._make_strat()
        assert strat.on_bar_update(_row()) is None

    def test_long_entry_when_below_lower_band_and_oversold(self):
        strat = self._make_strat()
        self._bypass_warmup(strat)
        # Price below lower band, RSI oversold
        row = _row(close=1970.0, bb_lower_20=1975.0, rsi_14=28.0, adx_14=15.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.meta["setup"] == "oversold"

    def test_short_entry_when_above_upper_band_and_overbought(self):
        strat = self._make_strat()
        self._bypass_warmup(strat)
        row = _row(close=2030.0, bb_upper_20=2025.0, rsi_14=72.0, adx_14=15.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.meta["setup"] == "overbought"

    def test_no_entry_when_adx_too_high(self):
        """Don't counter-trade into a strong trend."""
        strat = self._make_strat(adx_ceiling=30.0)
        self._bypass_warmup(strat)
        row = _row(close=1970.0, bb_lower_20=1975.0, rsi_14=28.0, adx_14=40.0)
        sig = strat.on_bar(row)
        assert sig is None

    def test_no_entry_when_rsi_not_oversold(self):
        """Price below band but RSI not oversold → no entry."""
        strat = self._make_strat(oversold_rsi=35.0)
        self._bypass_warmup(strat)
        row = _row(close=1970.0, bb_lower_20=1975.0, rsi_14=45.0, adx_14=15.0)
        sig = strat.on_bar(row)
        assert sig is None

    def test_no_entry_when_bands_too_tight(self):
        strat = self._make_strat(min_band_width=0.01)
        self._bypass_warmup(strat)
        row = _row(close=1970.0, bb_lower_20=1975.0, rsi_14=28.0,
                   adx_14=15.0, bb_width_20=0.005)  # below min_band_width
        sig = strat.on_bar(row)
        assert sig is None

    def test_stop_loss_below_entry_for_long(self):
        strat = self._make_strat(atr_multiplier=2.0)
        self._bypass_warmup(strat)
        row = _row(close=1970.0, bb_lower_20=1975.0, rsi_14=28.0,
                   adx_14=15.0, atr_14=15.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.stop_loss == pytest.approx(1970.0 - 2.0 * 15.0)

    def test_tp_is_bb_mid_for_long(self):
        strat = self._make_strat(use_mid_as_tp=True)
        self._bypass_warmup(strat)
        row = _row(close=1970.0, bb_lower_20=1975.0, bb_mid_20=2000.0,
                   rsi_14=28.0, adx_14=15.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.take_profit == pytest.approx(2000.0)

    def test_exit_when_price_reverts_to_mean(self):
        strat = self._make_strat()
        self._bypass_warmup(strat)
        strat._state.current_position = Direction.LONG
        # Price back at midline → exit
        row = _row(close=2001.0, bb_mid_20=2000.0, rsi_14=50.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.meta["exit_reason"] == "reverted_to_mean"

    def test_exit_after_max_holding_period(self):
        strat = self._make_strat(max_bars_in_trade=5)
        self._bypass_warmup(strat)
        strat._state.current_position = Direction.LONG
        strat._state.bars_in_trade = 5  # at the limit
        row = _row(close=1985.0, bb_mid_20=2000.0, rsi_14=40.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.meta["exit_reason"] == "max_holding_period"

    def test_reset_clears_all_state(self):
        strat = self._make_strat()
        strat._entry_price = 2000.0
        strat._stop_loss = 1985.0
        strat.reset()
        assert strat._entry_price is None
        assert strat._stop_loss is None
        assert strat._bar_count == 0


# ===========================================================================
# MLStrategy
# ===========================================================================

class MockLGBModel:
    """
    Stand-in for a real LightGBM model.
    Returns controllable probabilities for testing.
    """
    def __init__(self, proba: list[float]):
        # proba = [p_short, p_flat, p_long] for 3-class, or [p_up] for binary
        self._proba = np.array(proba, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return shape (1, 3) for 3-class
        return self._proba.reshape(1, -1)


FEATURE_COLS = ["rsi_14", "atr_14", "adx_14", "ema_9", "ema_21"]


class TestMLStrategy:
    def _make_strat(self, proba=(0.1, 0.3, 0.6), **kwargs) -> MLStrategy:
        model = MockLGBModel(list(proba))
        return MLStrategy(
            symbol="XAUUSD", timeframe="H1",
            model=model,
            feature_cols=FEATURE_COLS,
            long_threshold=0.55,
            short_threshold=0.55,
            exit_threshold=0.40,
            cooldown_bars=0,
            **kwargs,
        )

    def _bypass_warmup(self, strat: MLStrategy) -> None:
        strat._bar_count = strat.warmup_bars

    def test_is_ready_with_model_and_features(self):
        strat = self._make_strat()
        assert strat.is_ready

    def test_not_ready_without_model(self):
        strat = MLStrategy(symbol="XAUUSD", timeframe="H1",
                           feature_cols=FEATURE_COLS)
        assert not strat.is_ready

    def test_long_signal_when_p_long_exceeds_threshold(self):
        # p_long=0.70 > threshold=0.55 → LONG
        strat = self._make_strat(proba=[0.10, 0.20, 0.70])
        self._bypass_warmup(strat)
        sig = strat.on_bar(_row())
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.strength == pytest.approx(0.70, rel=1e-3)

    def test_short_signal_when_p_short_exceeds_threshold(self):
        # p_short=0.65 > threshold=0.55 → SHORT
        strat = self._make_strat(proba=[0.65, 0.25, 0.10])
        self._bypass_warmup(strat)
        sig = strat.on_bar(_row())
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.strength == pytest.approx(0.65, rel=1e-3)

    def test_no_signal_when_below_threshold(self):
        # All probabilities below threshold
        strat = self._make_strat(proba=[0.35, 0.35, 0.30])
        self._bypass_warmup(strat)
        sig = strat.on_bar(_row())
        assert sig is None

    def test_exit_when_confidence_drops(self):
        # Model confident in SHORT initially, then drops
        strat = self._make_strat(proba=[0.30, 0.30, 0.40])  # below thresholds
        self._bypass_warmup(strat)
        strat._state.current_position = Direction.SHORT
        strat._last_proba = (0.40, 0.30)  # p_short=0.30 < exit_threshold=0.40
        sig = strat.on_bar(_row())
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.meta["exit_reason"] == "confidence_dropped"

    def test_exit_after_max_bars(self):
        strat = self._make_strat(max_bars_in_trade=3)
        self._bypass_warmup(strat)
        strat._state.current_position = Direction.LONG
        strat._state.bars_in_trade = 3  # at limit
        sig = strat.on_bar(_row())
        assert sig is not None
        assert sig.direction == Direction.FLAT
        assert sig.meta["exit_reason"] == "max_holding_period"

    def test_missing_features_fallback_to_last_proba(self):
        strat = self._make_strat(proba=[0.10, 0.20, 0.70])
        self._bypass_warmup(strat)
        # Set a known last proba
        strat._last_proba = (0.70, 0.10)
        # Row missing required features
        row = _row()
        for col in FEATURE_COLS:
            row.pop(col, None)
        # Should fall back to last known proba → LONG
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_no_signal_before_warmup(self):
        strat = self._make_strat(proba=[0.05, 0.05, 0.90])
        # bar_count starts at 0 → on_bar_update suppresses
        sig = strat.on_bar_update(_row())
        assert sig is None

    def test_stop_and_tp_set_on_entry(self):
        strat = self._make_strat(
            proba=[0.10, 0.20, 0.70],
            atr_multiplier=1.5, tp_multiplier=3.0
        )
        self._bypass_warmup(strat)
        row = _row(close=2005.0, atr_14=20.0)
        sig = strat.on_bar(row)
        assert sig is not None
        assert sig.stop_loss == pytest.approx(2005.0 - 1.5 * 20.0)
        assert sig.take_profit == pytest.approx(2005.0 + 3.0 * 20.0)

    def test_reset_clears_proba_state(self):
        strat = self._make_strat()
        strat._last_proba = (0.9, 0.1)
        strat.reset()
        assert strat._last_proba is None
        assert strat._bar_count == 0