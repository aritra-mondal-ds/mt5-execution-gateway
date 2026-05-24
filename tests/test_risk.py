"""
tests/test_risk.py
------------------
Tests for the complete risk layer:
  - PositionSizer  : lot calculation, Kelly scaling, boundary clipping
  - DrawdownBreaker: daily/peak tracking, halt logic, daily reset
  - CostFilter     : cost estimation, edge/cost ratio, pass/reject
  - RiskEngine     : full evaluation pipeline, all rejection reasons

Run with:
    pytest tests/test_risk.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta, date

import pytest

from strategies.base_strategy import Direction, Signal
from risk.order import Order, OrderType, OrderStatus
from risk.position_sizer import PositionSizer, SizingResult
from risk.drawdown import DrawdownBreaker
from risk.cost_filter import CostFilter
from risk.engine import RiskEngine, AccountState, PositionInfo, RejectionReason


# ===========================================================================
# Shared helpers
# ===========================================================================

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _signal(
    symbol: str = "XAUUSD",
    direction: Direction = Direction.LONG,
    strength: float = 1.0,
    atr: float | None = 15.0,
    stop_loss: float | None = 1985.0,
    take_profit: float | None = 2030.0,
    strategy_id: str = "test_strat",
    meta: dict | None = None,
) -> Signal:
    return Signal(
        symbol=symbol,
        timeframe="H1",
        direction=direction,
        strength=strength,
        strategy_id=strategy_id,
        bar_time=_NOW,
        signal_time=_NOW,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr_at_signal=atr,
        meta=meta or {"close": 2000.0},
    )


def _account(
    equity: float = 10_000.0,
    positions: list[PositionInfo] | None = None,
) -> AccountState:
    return AccountState(
        equity=equity,
        balance=equity,
        open_positions=positions or [],
        daily_start_equity=equity,
        peak_equity=equity,
    )


def _position(symbol: str = "XAUUSD", lots: float = 0.1) -> PositionInfo:
    return PositionInfo(
        symbol=symbol,
        direction=Direction.LONG,
        lot_size=lots,
        entry_price=2000.0,
        strategy_id="test_strat",
        open_time=_NOW,
    )


def _engine(**kwargs) -> RiskEngine:
    defaults = dict(
        max_risk_per_trade=0.01,
        max_total_exposure=0.10,
        max_positions=5,
        atr_multiplier=1.5,
        tp_rr_ratio=2.0,
        max_daily_drawdown=0.03,
        max_peak_drawdown=0.08,
        min_edge_cost_ratio=0.0,   # disable cost filter by default for engine tests
        commission_per_lot=7.0,
        min_lot=0.01,
        max_lot=10.0,
    )
    defaults.update(kwargs)
    e = RiskEngine(**defaults)
    e.drawdown_breaker.initialize(equity=10_000.0)
    return e


# ===========================================================================
# Order dataclass
# ===========================================================================

class TestOrder:
    def test_symbol_delegates_to_signal(self):
        o = Order(signal=_signal(), lot_size=0.1)
        assert o.symbol == "XAUUSD"

    def test_direction_delegates_to_signal(self):
        o = Order(signal=_signal(direction=Direction.SHORT), lot_size=0.1)
        assert o.direction == Direction.SHORT

    def test_is_entry_for_long(self):
        o = Order(signal=_signal(direction=Direction.LONG), lot_size=0.1)
        assert o.is_entry
        assert not o.is_exit

    def test_is_exit_for_flat(self):
        o = Order(signal=_signal(direction=Direction.FLAT), lot_size=0.1)
        assert o.is_exit
        assert not o.is_entry

    def test_is_filled(self):
        o = Order(signal=_signal(), lot_size=0.1, status=OrderStatus.FILLED)
        assert o.is_filled

    def test_repr_contains_key_info(self):
        o = Order(signal=_signal(), lot_size=0.5)
        r = repr(o)
        assert "XAUUSD" in r
        assert "0.5" in r


# ===========================================================================
# PositionSizer
# ===========================================================================

class TestPositionSizer:
    def _sizer(self, **kwargs) -> PositionSizer:
        defaults = dict(max_risk_per_trade=0.01, atr_multiplier=1.5,
                        min_lot=0.01, max_lot=10.0)
        defaults.update(kwargs)
        return PositionSizer(**defaults)

    def test_basic_calculation(self):
        sizer = self._sizer()
        # equity=10000, risk=1%=100, stop=22.5pts, pip_value=10
        # lot = 100 / (22.5 * 10) = 0.44 → rounded to 0.44
        result = sizer.calculate(equity=10_000, atr=15.0, symbol="XAUUSD")
        assert result.lot_size > 0
        assert result.rejection_reason == ""

    def test_lot_size_scales_with_equity(self):
        sizer = self._sizer()
        r_small = sizer.calculate(equity=5_000, atr=15.0, symbol="XAUUSD")
        r_large = sizer.calculate(equity=20_000, atr=15.0, symbol="XAUUSD")
        assert r_large.lot_size > r_small.lot_size

    def test_lot_size_scales_inverse_with_atr(self):
        """Higher ATR → wider stop → smaller lot."""
        sizer = self._sizer()
        r_low_atr = sizer.calculate(equity=10_000, atr=10.0, symbol="XAUUSD")
        r_high_atr = sizer.calculate(equity=10_000, atr=30.0, symbol="XAUUSD")
        assert r_low_atr.lot_size > r_high_atr.lot_size

    def test_min_lot_clipping(self):
        """Very small equity → clips to min_lot."""
        sizer = self._sizer(min_lot=0.01)
        result = sizer.calculate(equity=10.0, atr=100.0, symbol="XAUUSD")
        assert result.lot_size == pytest.approx(0.01)
        assert result.clipped

    def test_max_lot_clipping(self):
        """Very large equity → clips to max_lot."""
        sizer = self._sizer(max_lot=1.0)
        result = sizer.calculate(equity=10_000_000, atr=1.0, symbol="XAUUSD")
        assert result.lot_size == pytest.approx(1.0)
        assert result.clipped

    def test_rejection_on_zero_equity(self):
        sizer = self._sizer()
        result = sizer.calculate(equity=0, atr=15.0, symbol="XAUUSD")
        assert result.rejection_reason == "zero_equity"
        assert result.lot_size == 0

    def test_rejection_on_missing_atr(self):
        sizer = self._sizer()
        result = sizer.calculate(equity=10_000, atr=None, symbol="XAUUSD")
        assert result.rejection_reason == "atr_missing"

    def test_stop_distance_override(self):
        """Explicit stop_distance overrides ATR-based calculation."""
        sizer = self._sizer()
        r_atr = sizer.calculate(equity=10_000, atr=15.0, symbol="XAUUSD")
        r_override = sizer.calculate(
            equity=10_000, atr=15.0, symbol="XAUUSD",
            stop_distance=15.0 * 1.5  # same as ATR-based
        )
        assert r_atr.lot_size == pytest.approx(r_override.lot_size, rel=1e-3)

    def test_kelly_reduces_position_size(self):
        """Kelly fraction < 1 should reduce lot vs no Kelly."""
        sizer_no_kelly = self._sizer(kelly_fraction=0.0)
        sizer_kelly = self._sizer(kelly_fraction=0.25)
        r_no = sizer_no_kelly.calculate(equity=10_000, atr=15.0, symbol="XAUUSD")
        r_k = sizer_kelly.calculate(
            equity=10_000, atr=15.0, symbol="XAUUSD",
            win_rate=0.55, avg_rr=2.0,
        )
        assert r_k.lot_size <= r_no.lot_size

    def test_kelly_with_negative_edge_gives_zero_scale(self):
        """Kelly with losing strategy → scale=0 → min_lot."""
        sizer = self._sizer(kelly_fraction=0.25, min_lot=0.01)
        result = sizer.calculate(
            equity=10_000, atr=15.0, symbol="XAUUSD",
            win_rate=0.30, avg_rr=1.0,  # negative edge
        )
        # Kelly scale should be 0 → lots clamped to min_lot
        assert result.lot_size == pytest.approx(0.01)

    def test_risk_fraction_reasonable(self):
        sizer = self._sizer(max_risk_per_trade=0.01)
        result = sizer.calculate(equity=10_000, atr=15.0, symbol="XAUUSD")
        # Risk fraction should be close to 1% (may differ due to rounding)
        assert result.risk_fraction < 0.05  # never more than 5%


# ===========================================================================
# DrawdownBreaker
# ===========================================================================

class TestDrawdownBreaker:
    def _breaker(self, as_of: datetime | None = None, **kwargs) -> DrawdownBreaker:
        defaults = dict(max_daily_drawdown=0.03, max_peak_drawdown=0.08)
        defaults.update(kwargs)
        b = DrawdownBreaker(**defaults)
        b.initialize(equity=10_000.0, as_of=as_of)
        return b

    def test_not_halted_initially(self):
        b = self._breaker()
        assert not b.is_halted

    def test_halted_before_initialize(self):
        """Uninitialized breaker returns is_halted=True for safety."""
        b = DrawdownBreaker()
        assert b.is_halted

    def test_no_halt_within_limits(self):
        b = self._breaker()
        halted = b.update(equity=9_800.0)  # 2% daily loss, under 3% limit
        assert not halted
        assert not b.is_halted

    def test_daily_halt_at_limit(self):
        b = self._breaker(max_daily_drawdown=0.03)
        halted = b.update(equity=9_700.0)  # exactly 3% loss
        assert halted
        assert b.is_halted
        assert "daily" in b.state.halt_reason

    def test_peak_halt_at_limit(self):
        b = self._breaker(max_peak_drawdown=0.08)
        b.update(equity=11_000.0)   # new peak
        halted = b.update(equity=10_120.0)  # 8% from peak of 11000
        assert halted
        assert b.is_halted
        assert "peak" in b.state.halt_reason

    def test_daily_drawdown_tracked_correctly(self):
        b = self._breaker()
        b.update(equity=9_500.0)  # 5% loss
        assert b.daily_drawdown == pytest.approx(0.05, rel=1e-3)

    def test_peak_updates_on_new_high(self):
        b = self._breaker()
        b.update(equity=11_000.0)
        assert b.state.peak_equity == pytest.approx(11_000.0)

    def test_manual_resume(self):
        b = self._breaker(max_daily_drawdown=0.03)
        b.update(equity=9_700.0)  # trigger halt
        assert b.is_halted
        b.resume()
        assert not b.is_halted

    def test_daily_reset_on_new_day(self):
        # Initialize with day1's date so daily_start_date = 2024-06-01.
        # day1 update: same date → no reset, drawdown calculated → halt.
        # day2 update: new date → _daily_reset fires → halt lifted.
        init_day = datetime(2024, 6, 1, tzinfo=timezone.utc)
        b = self._breaker(max_daily_drawdown=0.03, as_of=init_day)
        day1 = datetime(2024, 6, 1, 23, 59, tzinfo=timezone.utc)
        day2 = datetime(2024, 6, 2, 0,  1, tzinfo=timezone.utc)

        b.update(equity=9_700.0, as_of=day1)  # trigger daily halt
        assert b.is_halted

        # New day → daily halt auto-lifts
        b.update(equity=9_700.0, as_of=day2)
        assert not b.is_halted

    def test_peak_halt_not_lifted_by_daily_reset(self):
        """Peak drawdown halt persists across days."""
        b = self._breaker(max_peak_drawdown=0.08)
        day1 = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
        day2 = datetime(2024, 6, 2, 12, tzinfo=timezone.utc)

        b.update(equity=11_000.0, as_of=day1)  # new peak
        b.update(equity=10_120.0, as_of=day1)  # 8% peak dd → halt
        assert b.is_halted

        b.update(equity=10_120.0, as_of=day2)  # new day
        assert b.is_halted  # still halted — peak dd persists

    def test_daily_drawdown_resets_after_new_day(self):
        init_day = datetime(2024, 6, 1, tzinfo=timezone.utc)
        b = self._breaker(as_of=init_day)
        day1 = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
        day2 = datetime(2024, 6, 2, 12, tzinfo=timezone.utc)
        b.update(equity=9_500.0, as_of=day1)
        assert b.daily_drawdown > 0
        b.update(equity=9_500.0, as_of=day2)
        # After reset: daily_start_equity = 9500, equity = 9500 → daily_dd = 0
        assert b.daily_drawdown == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# CostFilter
# ===========================================================================

class TestCostFilter:
    def _filter(self, **kwargs) -> CostFilter:
        defaults = dict(
            min_edge_cost_ratio=1.5,
            commission_per_lot=7.0,
            slippage_factor=0.5,
        )
        defaults.update(kwargs)
        return CostFilter(**defaults)

    def test_passes_when_ratio_above_threshold(self):
        f = self._filter(min_edge_cost_ratio=1.5)
        # Large stop + high RR → big edge
        result = f.evaluate(
            symbol="XAUUSD", lot_size=0.1,
            stop_distance=50.0, rr_ratio=3.0, signal_strength=1.0,
        )
        assert result.passed

    def test_rejects_when_ratio_below_threshold(self):
        f = self._filter(min_edge_cost_ratio=1.5)
        # Tiny stop → tiny edge, but full cost
        result = f.evaluate(
            symbol="XAUUSD", lot_size=1.0,
            stop_distance=1.0, rr_ratio=1.0, signal_strength=0.5,
        )
        assert not result.passed
        assert result.rejection_reason != ""

    def test_exit_signal_always_passes(self):
        """stop_distance=None means exit → skip filter."""
        f = self._filter(min_edge_cost_ratio=999.0)
        result = f.evaluate(
            symbol="XAUUSD", lot_size=0.1,
            stop_distance=None,
        )
        assert result.passed
        assert result.ratio == pytest.approx(999.0)

    def test_cost_components_sum_correctly(self):
        f = self._filter(commission_per_lot=7.0, slippage_factor=0.5)
        result = f.evaluate(
            symbol="XAUUSD", lot_size=1.0,
            stop_distance=20.0, rr_ratio=2.0,
        )
        expected_total = (
            result.spread_cost +
            result.commission_cost +
            result.slippage_cost
        )
        assert result.total_cost == pytest.approx(expected_total, rel=1e-6)

    def test_signal_strength_scales_edge(self):
        """Lower confidence → lower edge estimate → harder to pass."""
        f = self._filter(min_edge_cost_ratio=1.5)
        r_high = f.evaluate("XAUUSD", 0.1, 20.0, rr_ratio=2.0, signal_strength=1.0)
        r_low  = f.evaluate("XAUUSD", 0.1, 20.0, rr_ratio=2.0, signal_strength=0.3)
        assert r_high.edge_estimate > r_low.edge_estimate

    def test_larger_lot_increases_both_cost_and_edge(self):
        f = self._filter()
        r_small = f.evaluate("XAUUSD", 0.1, 20.0, rr_ratio=2.0)
        r_large = f.evaluate("XAUUSD", 1.0, 20.0, rr_ratio=2.0)
        assert r_large.total_cost > r_small.total_cost
        assert r_large.edge_estimate > r_small.edge_estimate
        # Ratio should be the same (both scale linearly with lot_size)
        assert r_large.ratio == pytest.approx(r_small.ratio, rel=1e-3)


# ===========================================================================
# RiskEngine
# ===========================================================================

class TestRiskEngine:
    def test_entry_signal_approved(self):
        engine = _engine()
        result = engine.evaluate(_signal(), _account())
        assert result.approved
        assert result.order is not None
        assert result.order.lot_size > 0

    def test_exit_signal_always_approved(self):
        """Exit bypasses all entry checks."""
        engine = _engine()
        engine.drawdown_breaker._state.is_halted = True  # halt active
        sig = _signal(direction=Direction.FLAT)
        result = engine.evaluate(sig, _account())
        assert result.approved

    def test_rejected_when_drawdown_halted(self):
        engine = _engine()
        engine.drawdown_breaker._state.is_halted = True
        engine.drawdown_breaker._state.halt_reason = "daily_drawdown_limit"
        result = engine.evaluate(_signal(), _account())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.DRAWDOWN_HALT

    def test_rejected_when_already_positioned(self):
        engine = _engine()
        # System position (strategy_id does NOT start with mt5_ticket_) blocks re-entry
        sys_pos = PositionInfo(
            symbol="XAUUSD", direction=Direction.LONG,
            lot_size=0.1, entry_price=2000.0,
            strategy_id="trend_ema9_21",   # system position
            open_time=_NOW,
        )
        account = _account(positions=[sys_pos])
        result = engine.evaluate(_signal(symbol="XAUUSD"), account)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.ALREADY_POSITIONED

    def test_external_position_does_not_block_system_entry(self):
        """Positions opened manually in MT5 (mt5_ticket_ prefix) should NOT
        block the system from trading the same symbol."""
        engine = _engine()
        ext_pos = PositionInfo(
            symbol="XAUUSD", direction=Direction.LONG,
            lot_size=0.1, entry_price=2000.0,
            strategy_id="mt5_ticket_123456",  # external position
            open_time=_NOW,
        )
        account = _account(positions=[ext_pos])
        result = engine.evaluate(_signal(symbol="XAUUSD"), account)
        assert result.approved

    def test_rejected_when_max_positions_reached(self):
        engine = _engine(max_positions=2)
        account = _account(positions=[
            _position("EURUSD"), _position("GBPUSD"),
        ])
        result = engine.evaluate(_signal(symbol="XAUUSD"), account)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.EXPOSURE_LIMIT

    def test_rejected_when_atr_missing(self):
        engine = _engine()
        sig = _signal(atr=None, stop_loss=None)
        result = engine.evaluate(sig, _account())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.SIZING_FAILED

    def test_rejected_by_cost_filter(self):
        engine = _engine(min_edge_cost_ratio=999.0)  # impossible threshold
        result = engine.evaluate(_signal(), _account())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.COST_FILTER

    def test_order_has_stop_loss(self):
        engine = _engine()
        result = engine.evaluate(_signal(stop_loss=1985.0), _account())
        assert result.approved
        assert result.order.stop_loss == pytest.approx(1985.0)

    def test_order_has_take_profit(self):
        engine = _engine()
        result = engine.evaluate(_signal(take_profit=2030.0), _account())
        assert result.approved
        assert result.order.take_profit == pytest.approx(2030.0)

    def test_order_lot_size_positive(self):
        engine = _engine()
        result = engine.evaluate(_signal(), _account())
        assert result.approved
        assert result.order.lot_size >= 0.01

    def test_order_lot_size_respects_equity(self):
        """Smaller equity → smaller lot size."""
        engine = _engine()
        r_small = engine.evaluate(_signal(), _account(equity=1_000))
        r_large = engine.evaluate(_signal(), _account(equity=100_000))
        # Re-init drawdown breaker for second call
        engine.drawdown_breaker.initialize(equity=100_000)
        r_large = engine.evaluate(_signal(), _account(equity=100_000))
        assert r_large.order.lot_size >= r_small.order.lot_size

    def test_evaluation_result_repr(self):
        engine = _engine()
        result = engine.evaluate(_signal(), _account())
        r = repr(result)
        assert "APPROVED" in r or "REJECTED" in r

    def test_audit_log_populated(self):
        engine = _engine()
        engine.evaluate(_signal(), _account())
        assert len(engine.audit_log) == 1

    def test_rejection_stats_counted(self):
        engine = _engine()
        # Trigger a halt rejection
        engine.drawdown_breaker._state.is_halted = True
        engine.drawdown_breaker._state.halt_reason = "daily_drawdown_limit"
        engine.evaluate(_signal(), _account())
        stats = engine.rejection_stats()
        assert stats.get("drawdown_halt", 0) == 1

    def test_short_signal_approved(self):
        engine = _engine()
        sig = _signal(direction=Direction.SHORT, stop_loss=2015.0, take_profit=1970.0)
        result = engine.evaluate(sig, _account())
        assert result.approved
        assert result.order.direction == Direction.SHORT

    def test_exit_uses_existing_position_lot_size(self):
        """Exit order lot size matches the open position."""
        engine = _engine()
        pos = _position("XAUUSD", lots=0.5)
        account = _account(positions=[pos])
        sig = _signal(direction=Direction.FLAT)
        result = engine.evaluate(sig, account)
        assert result.approved
        assert result.order.lot_size == pytest.approx(0.5)

    def test_multiple_symbols_allowed(self):
        """Different symbols can hold simultaneous positions."""
        engine = _engine(max_positions=5)
        account = _account(positions=[_position("EURUSD")])
        result = engine.evaluate(_signal(symbol="XAUUSD"), account)
        assert result.approved

    def test_drawdown_update_triggers_halt(self):
        """Updating drawdown breaker mid-session halts further entries."""
        engine = _engine(max_daily_drawdown=0.03)
        engine.drawdown_breaker.update(equity=9_700.0)  # 3% loss → halt
        result = engine.evaluate(_signal(), _account())
        assert not result.approved
        assert result.rejection_reason == RejectionReason.DRAWDOWN_HALT