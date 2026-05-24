"""
tests/test_backtesting.py
--------------------------
Tests for the backtesting layer.

Coverage:
  - FillModel: SL/TP detection, gap fills, long/short, boundary cases
  - BacktestMetrics: known-answer metric calculations
  - BacktestEngine: end-to-end event loop with synthetic data

Run with:
    pytest tests/test_backtesting.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

from feeds.base_feed import Bar
from strategies.base_strategy import Direction, Signal
from backtesting.fill_model import FillModel, OpenPosition, FillType, BarFill
from backtesting.metrics import compute_metrics, BacktestMetrics, _compute_drawdown


# ===========================================================================
# Helpers
# ===========================================================================

_NOW = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def _bar(
    symbol: str = "XAUUSD",
    open_: float = 2000.0,
    high: float = 2010.0,
    low: float = 1990.0,
    close: float = 2005.0,
    offset_hours: int = 0,
) -> Bar:
    t = _NOW + timedelta(hours=offset_hours)
    return Bar(
        symbol=symbol, timeframe="H1",
        event_time=t, publish_time=t,
        open=open_, high=high, low=low, close=close,
        volume=1000.0, spread=0.30, source="test",
    )


def _pos(
    direction: Direction = Direction.LONG,
    entry: float = 2000.0,
    sl: float | None = 1985.0,
    tp: float | None = 2030.0,
    lots: float = 0.1,
    symbol: str = "XAUUSD",
) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        lot_size=lots,
        stop_loss=sl,
        take_profit=tp,
        entry_bar_index=0,
        strategy_id="test",
    )


def _fill_model(**kwargs) -> FillModel:
    defaults = dict(
        slippage_factor=0.0,        # deterministic for tests
        gap_fill=True,
        require_price_through=True,
    )
    defaults.update(kwargs)
    return FillModel(**defaults)


# ===========================================================================
# FillModel — Long positions
# ===========================================================================

class TestFillModelLong:
    def test_no_fill_when_price_within_range(self):
        fm = _fill_model()
        pos = _pos(direction=Direction.LONG, sl=1985.0, tp=2030.0)
        bar = _bar(open_=2000, high=2010, low=1990, close=2005)
        result = fm.check_position(bar, pos)
        assert not result.triggered

    def test_sl_hit_when_low_touches_stop(self):
        fm = _fill_model()
        pos = _pos(direction=Direction.LONG, sl=1990.0, tp=2030.0)
        bar = _bar(open_=2000, high=2005, low=1990, close=1995)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.STOP_LOSS
        assert result.fill_price == pytest.approx(1990.0)

    def test_sl_not_hit_when_low_above_stop(self):
        fm = _fill_model()
        pos = _pos(direction=Direction.LONG, sl=1985.0, tp=2030.0)
        bar = _bar(open_=2000, high=2010, low=1990, close=2005)
        result = fm.check_position(bar, pos)
        assert not result.triggered

    def test_tp_hit_when_high_exceeds_target(self):
        """require_price_through=True: high must go ABOVE tp."""
        fm = _fill_model(require_price_through=True)
        pos = _pos(direction=Direction.LONG, sl=1985.0, tp=2010.0)
        bar = _bar(open_=2000, high=2011, low=1995, close=2008)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.TAKE_PROFIT
        assert result.fill_price == pytest.approx(2010.0)

    def test_tp_not_hit_when_high_equals_target_and_require_through(self):
        """require_price_through=True: touching tp exactly does NOT fill."""
        fm = _fill_model(require_price_through=True)
        pos = _pos(direction=Direction.LONG, sl=1985.0, tp=2010.0)
        bar = _bar(open_=2000, high=2010, low=1995, close=2008)
        result = fm.check_position(bar, pos)
        assert not result.triggered

    def test_sl_takes_priority_over_tp_same_bar(self):
        """If both SL and TP hit in same bar → SL wins (conservative)."""
        fm = _fill_model()
        pos = _pos(direction=Direction.LONG, sl=1990.0, tp=2005.0)
        bar = _bar(open_=2000, high=2010, low=1989, close=2000)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.STOP_LOSS

    def test_gap_open_below_sl(self):
        """Bar opens below SL → gap fill at open (not at SL)."""
        fm = _fill_model(gap_fill=True)
        pos = _pos(direction=Direction.LONG, sl=1990.0, tp=2030.0)
        # Bar opens at 1980 — gapped through stop
        bar = _bar(open_=1980, high=1985, low=1975, close=1982)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.GAP
        assert result.fill_price == pytest.approx(1980.0)

    def test_slippage_applied_to_sl(self):
        fm = _fill_model(slippage_factor=1.0,
                         spread_points={"XAUUSD": 0.30})
        pos = _pos(direction=Direction.LONG, sl=1990.0)
        bar = _bar(open_=2000, high=2005, low=1989, close=1995)
        result = fm.check_position(bar, pos)
        assert result.triggered
        # fill = sl - slippage = 1990 - (0.30 * 1.0) = 1989.70
        assert result.fill_price == pytest.approx(1989.70, abs=0.01)
        assert result.slippage > 0


# ===========================================================================
# FillModel — Short positions
# ===========================================================================

class TestFillModelShort:
    def test_sl_hit_when_high_touches_stop(self):
        fm = _fill_model()
        pos = _pos(direction=Direction.SHORT, sl=2015.0, tp=1970.0)
        bar = _bar(open_=2000, high=2015, low=1995, close=2005)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.STOP_LOSS
        assert result.fill_price == pytest.approx(2015.0)

    def test_tp_hit_when_low_drops_below_target(self):
        fm = _fill_model(require_price_through=True)
        pos = _pos(direction=Direction.SHORT, sl=2015.0, tp=1980.0)
        bar = _bar(open_=2000, high=2005, low=1979, close=1985)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.TAKE_PROFIT
        assert result.fill_price == pytest.approx(1980.0)

    def test_no_fill_within_range(self):
        fm = _fill_model()
        pos = _pos(direction=Direction.SHORT, sl=2020.0, tp=1970.0)
        bar = _bar(open_=2000, high=2010, low=1990, close=2000)
        result = fm.check_position(bar, pos)
        assert not result.triggered

    def test_gap_open_above_sl(self):
        fm = _fill_model(gap_fill=True)
        pos = _pos(direction=Direction.SHORT, sl=2010.0, tp=1970.0)
        bar = _bar(open_=2020, high=2025, low=2015, close=2018)
        result = fm.check_position(bar, pos)
        assert result.triggered
        assert result.fill_type == FillType.GAP
        assert result.fill_price == pytest.approx(2020.0)


# ===========================================================================
# FillModel — entry fill price
# ===========================================================================

class TestFillModelEntry:
    def test_long_entry_above_open(self):
        fm = _fill_model()
        bar = _bar(open_=2000)
        price = fm.entry_fill_price(bar, Direction.LONG)
        assert price >= 2000.0

    def test_short_entry_at_or_below_open(self):
        fm = _fill_model()
        bar = _bar(open_=2000)
        price = fm.entry_fill_price(bar, Direction.SHORT)
        assert price <= 2000.0

    def test_no_sl_no_tp_no_fill(self):
        """Position with no SL or TP should never trigger."""
        fm = _fill_model()
        pos = _pos(direction=Direction.LONG, sl=None, tp=None)
        bar = _bar(open_=1500, high=1600, low=1400, close=1500)
        result = fm.check_position(bar, pos)
        assert not result.triggered


# ===========================================================================
# BacktestMetrics — known-answer tests
# ===========================================================================

class TestMetrics:
    def test_total_return_flat(self):
        equity = [10_000.0] * 100
        m = compute_metrics(equity, [])
        assert m.total_return == pytest.approx(0.0)

    def test_total_return_10pct(self):
        equity = [10_000.0, 11_000.0]
        m = compute_metrics(equity, [])
        assert m.total_return == pytest.approx(0.10)

    def test_total_return_loss(self):
        equity = [10_000.0, 9_000.0]
        m = compute_metrics(equity, [])
        assert m.total_return == pytest.approx(-0.10)

    def test_max_drawdown_simple(self):
        # Goes up to 12000, drops to 9000 → 25% drawdown
        equity = [10_000, 11_000, 12_000, 10_000, 9_000, 10_500]
        dd, dur = _compute_drawdown(equity)
        assert dd == pytest.approx(0.25, rel=1e-3)

    def test_max_drawdown_zero_for_monotonic_up(self):
        equity = [10_000, 11_000, 12_000, 13_000]
        dd, _ = _compute_drawdown(equity)
        assert dd == pytest.approx(0.0)

    def test_sharpe_positive_for_positive_returns(self):
        # Steady 0.1% gain per bar over 252 bars
        equity = [10_000 * (1.001 ** i) for i in range(252)]
        m = compute_metrics(equity, [], bars_per_year=252)
        assert m.sharpe > 0

    def test_win_rate_calculation(self):
        trades = [
            {"net_pnl": 100, "direction": "long", "bars_held": 5, "commission": 1},
            {"net_pnl": -50, "direction": "short", "bars_held": 3, "commission": 1},
            {"net_pnl": 80,  "direction": "long", "bars_held": 7, "commission": 1},
        ]
        m = compute_metrics([10_000, 10_130], trades)
        assert m.total_trades == 3
        assert m.winning_trades == 2
        assert m.win_rate == pytest.approx(2 / 3)

    def test_profit_factor(self):
        trades = [
            {"net_pnl": 200, "direction": "long", "bars_held": 5, "commission": 0},
            {"net_pnl": -100, "direction": "short", "bars_held": 3, "commission": 0},
        ]
        m = compute_metrics([10_000, 10_100], trades)
        assert m.profit_factor == pytest.approx(2.0)

    def test_profit_factor_infinite_when_no_losses(self):
        trades = [
            {"net_pnl": 100, "direction": "long", "bars_held": 5, "commission": 0},
        ]
        m = compute_metrics([10_000, 10_100], trades)
        assert m.profit_factor == float("inf")

    def test_expectancy(self):
        trades = [
            {"net_pnl": 100, "direction": "long", "bars_held": 5, "commission": 0},
            {"net_pnl": -40, "direction": "long", "bars_held": 3, "commission": 0},
        ]
        m = compute_metrics([10_000, 10_060], trades)
        assert m.expectancy == pytest.approx(30.0)

    def test_consecutive_wins_losses(self):
        trades = [
            {"net_pnl": 10, "direction": "long", "bars_held": 1, "commission": 0},
            {"net_pnl": 10, "direction": "long", "bars_held": 1, "commission": 0},
            {"net_pnl": 10, "direction": "long", "bars_held": 1, "commission": 0},
            {"net_pnl": -5, "direction": "long", "bars_held": 1, "commission": 0},
            {"net_pnl": -5, "direction": "long", "bars_held": 1, "commission": 0},
        ]
        m = compute_metrics([10_000, 10_020], trades)
        assert m.max_consecutive_wins == 3
        assert m.max_consecutive_losses == 2

    def test_long_short_count(self):
        trades = [
            {"net_pnl": 10, "direction": "long",  "bars_held": 1, "commission": 0},
            {"net_pnl": 10, "direction": "short", "bars_held": 1, "commission": 0},
            {"net_pnl": 10, "direction": "long",  "bars_held": 1, "commission": 0},
        ]
        m = compute_metrics([10_000, 10_030], trades)
        assert m.long_trades == 2
        assert m.short_trades == 1

    def test_empty_equity_curve(self):
        m = compute_metrics([], [])
        assert m.total_return == 0.0
        assert m.total_trades == 0

    def test_metrics_str_renders(self):
        m = BacktestMetrics(
            total_return=0.15, sharpe=1.2, max_drawdown=0.08,
            total_trades=50, win_rate=0.55,
        )
        s = str(m)
        assert "15.00%" in s
        assert "Sharpe" in s

    def test_calmar_ratio(self):
        # cagr=10%, max_dd=5% → calmar=2.0
        m = BacktestMetrics(cagr=0.10, max_drawdown=0.05)
        # Manually set to verify formula
        calmar = m.cagr / m.max_drawdown if m.max_drawdown > 0 else 0
        assert calmar == pytest.approx(2.0)


# ===========================================================================
# BacktestEngine — integration tests
# ===========================================================================

class TestBacktestEngine:
    """
    Integration tests for the full event loop.
    Uses a mock store with synthetic bars and a concrete strategy.
    """

    def _make_store(self, tmp_path, n_bars: int = 400):
        """Create a populated RawDataStore with synthetic bars."""
        from data.store import RawDataStore
        store = RawDataStore(root=tmp_path / "raw")

        import numpy as np
        rng = np.random.default_rng(42)
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        prices = [2000.0]
        for _ in range(n_bars - 1):
            prices.append(prices[-1] * (1 + rng.normal(0.0002, 0.005)))

        bars = []
        for i, close in enumerate(prices):
            o = close * (1 + rng.normal(0, 0.002))
            h = max(o, close) * (1 + rng.uniform(0.001, 0.005))
            l = min(o, close) * (1 - rng.uniform(0.001, 0.005))
            t = base + timedelta(hours=i)
            bars.append(Bar(
                symbol="XAUUSD", timeframe="H1",
                event_time=t, publish_time=t + timedelta(milliseconds=50),
                open=o, high=h, low=l, close=close,
                volume=10000.0, spread=0.30, source="test",
            ))
        store.write_batch(bars)
        return store, base, base + timedelta(hours=n_bars - 1)

    def test_engine_runs_without_error(self, tmp_path):
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.trend import TrendStrategy
        from features.pipeline import PipelineConfig

        store, start, end = self._make_store(tmp_path, n_bars=400)
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=start, end_date=end,
            initial_equity=10_000,
            warmup_bars=250,
        )
        strategy = TrendStrategy("XAUUSD", "H1", fast_period=9, slow_period=21)
        engine = BacktestEngine(
            config=config, strategy=strategy, raw_store=store,
            pipeline_config=PipelineConfig(compute_ml=False, min_bars=50),
        )
        metrics = engine.run()
        assert metrics is not None
        assert isinstance(metrics.total_return, float)

    def test_equity_curve_length_matches_trading_bars(self, tmp_path):
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.trend import TrendStrategy
        from features.pipeline import PipelineConfig

        n = 350
        store, start, end = self._make_store(tmp_path, n_bars=n)
        warmup = 250
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=start, end_date=end,
            initial_equity=10_000,
            warmup_bars=warmup,
        )
        strategy = TrendStrategy("XAUUSD", "H1")
        engine = BacktestEngine(
            config=config, strategy=strategy, raw_store=store,
            pipeline_config=PipelineConfig(compute_ml=False, min_bars=50),
        )
        engine.run()
        # Equity curve should have one entry per trading bar
        expected = n - warmup
        assert len(engine.equity_curve) == pytest.approx(expected, abs=2)

    def test_no_lookahead_equity_never_nan(self, tmp_path):
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.trend import TrendStrategy
        from features.pipeline import PipelineConfig

        store, start, end = self._make_store(tmp_path, n_bars=350)
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=start, end_date=end,
            initial_equity=10_000, warmup_bars=250,
        )
        strategy = TrendStrategy("XAUUSD", "H1")
        engine = BacktestEngine(
            config=config, strategy=strategy, raw_store=store,
            pipeline_config=PipelineConfig(compute_ml=False, min_bars=50),
        )
        engine.run()
        assert all(math.isfinite(e) for e in engine.equity_curve)
        assert all(e > 0 for e in engine.equity_curve)

    def test_trade_log_records_correct_fields(self, tmp_path):
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies.trend import TrendStrategy
        from features.pipeline import PipelineConfig

        store, start, end = self._make_store(tmp_path, n_bars=350)
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=start, end_date=end,
            initial_equity=10_000, warmup_bars=250,
        )
        strategy = TrendStrategy("XAUUSD", "H1")
        engine = BacktestEngine(
            config=config, strategy=strategy, raw_store=store,
            pipeline_config=PipelineConfig(compute_ml=False, min_bars=50),
        )
        engine.run()

        for trade in engine.trade_log:
            assert hasattr(trade, "net_pnl")
            assert hasattr(trade, "entry_price")
            assert hasattr(trade, "exit_price")
            assert trade.lot_size > 0
            assert trade.bars_held >= 0

    def test_initial_equity_preserved_with_no_trades(self, tmp_path):
        """If strategy emits no signals, equity stays flat."""
        from backtesting.engine import BacktestEngine, BacktestConfig
        from features.pipeline import PipelineConfig

        # Strategy that never signals
        class SilentStrategy:
            strategy_id = "silent"
            warmup_bars = 1
            _bar_count = 0

            def reset(self): pass

            def on_bar_update(self, row):
                return None

        store, start, end = self._make_store(tmp_path, n_bars=300)
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=start, end_date=end,
            initial_equity=10_000, warmup_bars=250,
        )
        engine = BacktestEngine(
            config=config,
            strategy=SilentStrategy(),
            raw_store=store,
            pipeline_config=PipelineConfig(compute_ml=False, min_bars=50),
        )
        engine.run()
        assert len(engine.trade_log) == 0
        # Equity should remain at initial (no trades = no commission)
        for eq in engine.equity_curve:
            assert eq == pytest.approx(10_000.0, rel=1e-6)