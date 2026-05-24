"""
tests/test_execution.py
------------------------
Tests for the execution layer.

Coverage:
  - BrokerBase interface contract (via ConcreteTestBroker)
  - PaperBroker: connect/disconnect, price updates, market order fills,
    position opening/closing, P&L calculation, slippage, analytics
  - MT5Broker is NOT tested here (requires live terminal).
    It is validated by integration tests run manually.
  - Order/FillResult lifecycle

Run with:
    pytest tests/test_execution.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from execution.broker_base import (
    BrokerBase, BrokerPosition, BrokerAccountInfo,
    FillResult, BrokerError,
)
from execution.paper_broker import PaperBroker
from risk.order import Order, OrderStatus, OrderType
from strategies.base_strategy import Direction, Signal


# ===========================================================================
# Shared helpers
# ===========================================================================

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _signal(
    symbol: str = "XAUUSD",
    direction: Direction = Direction.LONG,
    stop_loss: float | None = 1985.0,
    take_profit: float | None = 2030.0,
    strategy_id: str = "test",
) -> Signal:
    return Signal(
        symbol=symbol, timeframe="H1",
        direction=direction, strength=1.0,
        strategy_id=strategy_id,
        bar_time=_NOW, signal_time=_NOW,
        stop_loss=stop_loss, take_profit=take_profit,
        atr_at_signal=15.0,
    )


def _order(
    symbol: str = "XAUUSD",
    direction: Direction = Direction.LONG,
    lot_size: float = 0.1,
    stop_loss: float | None = 1985.0,
    take_profit: float | None = 2030.0,
) -> Order:
    return Order(
        signal=_signal(symbol=symbol, direction=direction,
                       stop_loss=stop_loss, take_profit=take_profit),
        lot_size=lot_size,
        order_type=OrderType.MARKET,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def _broker(
    equity: float = 10_000.0,
    seed: int = 42,
    commission: float = 0.0,   # default 0 for simpler P&L assertions
    slippage: float = 0.0,     # default 0 for deterministic fill prices
) -> PaperBroker:
    b = PaperBroker(
        initial_equity=equity,
        commission_per_lot=commission,
        slippage_std_factor=slippage,
        seed=seed,
    )
    b.connect()
    b.update_price("XAUUSD", bid=1999.70, ask=2000.30)
    b.update_price("EURUSD", bid=1.08490, ask=1.08510)
    return b


# ===========================================================================
# BrokerBase interface contract
# ===========================================================================

class ConcreteBroker(BrokerBase):
    """Minimal concrete implementation for interface testing."""

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_account_info(self) -> BrokerAccountInfo:
        return BrokerAccountInfo(
            equity=10_000, balance=10_000,
            margin_used=0, margin_free=10_000, profit=0,
        )

    def get_positions(self) -> list[BrokerPosition]:
        return []

    def submit(self, order: Order) -> str:
        order.order_id = "test_ticket"
        order.status = OrderStatus.SUBMITTED
        self.track(order)
        return "test_ticket"

    def cancel(self, order_id: str) -> bool:
        return True

    def get_fill(self, order_id: str) -> FillResult:
        return FillResult(order_id=order_id, status=OrderStatus.FILLED,
                          fill_price=2000.0, fill_time=self.utcnow())

    def close_position(self, symbol: str, lot_size=None) -> FillResult:
        return FillResult(order_id="", status=OrderStatus.FILLED,
                          fill_price=2001.0, fill_time=self.utcnow())


class TestBrokerBase:
    def test_context_manager_connects_disconnects(self):
        b = ConcreteBroker("test")
        with b:
            assert b.is_connected
        assert not b.is_connected

    def test_track_stores_order(self):
        b = ConcreteBroker("test")
        order = _order()
        b.connect()
        b.submit(order)
        assert b.get_tracked("test_ticket") is order

    def test_utcnow_is_aware(self):
        b = ConcreteBroker("test")
        t = b.utcnow()
        assert t.tzinfo is not None

    def test_broker_error_has_code(self):
        err = BrokerError("test error", code=500)
        assert err.code == 500
        assert "test error" in str(err)


# ===========================================================================
# PaperBroker — connection and price management
# ===========================================================================

class TestPaperBrokerConnection:
    def test_connect_sets_connected(self):
        b = PaperBroker(initial_equity=10_000)
        assert not b.is_connected
        b.connect()
        assert b.is_connected

    def test_disconnect_clears_connected(self):
        b = PaperBroker()
        b.connect()
        b.disconnect()
        assert not b.is_connected

    def test_initial_equity(self):
        b = _broker(equity=25_000)
        info = b.get_account_info()
        assert info.equity == pytest.approx(25_000)
        assert info.balance == pytest.approx(25_000)

    def test_submit_without_connect_raises(self):
        b = PaperBroker()
        with pytest.raises(BrokerError, match="not connected"):
            b.submit(_order())

    def test_submit_without_price_raises(self):
        b = PaperBroker()
        b.connect()
        # No price set for XAUUSD → should raise
        with pytest.raises(BrokerError):
            b.submit(_order())

    def test_update_price_from_bar(self):
        b = PaperBroker()
        b.connect()
        b.update_price_from_bar("XAUUSD", close=2000.0, spread=0.5)
        bid, ask = b._prices["XAUUSD"]
        assert ask > bid
        assert ask == pytest.approx(2000.25)
        assert bid == pytest.approx(1999.75)


# ===========================================================================
# PaperBroker — order submission and fills
# ===========================================================================

class TestPaperBrokerOrders:
    def test_market_order_returns_ticket(self):
        b = _broker()
        ticket = b.submit(_order())
        assert isinstance(ticket, str)
        assert len(ticket) > 0

    def test_market_long_fills_at_ask(self):
        """Long market order fills at ask price (+ zero slippage)."""
        b = _broker(slippage=0.0)
        b.update_price("XAUUSD", bid=1999.70, ask=2000.30)
        order = _order(direction=Direction.LONG)
        b.submit(order)
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == pytest.approx(2000.30, abs=0.01)

    def test_market_short_fills_at_bid(self):
        """Short market order fills at bid price (+ zero slippage)."""
        b = _broker(slippage=0.0)
        b.update_price("XAUUSD", bid=1999.70, ask=2000.30)
        order = _order(direction=Direction.SHORT)
        b.submit(order)
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == pytest.approx(1999.70, abs=0.01)

    def test_fill_time_is_utc_aware(self):
        b = _broker()
        order = _order()
        b.submit(order)
        assert order.fill_time is not None
        assert order.fill_time.tzinfo is not None

    def test_get_fill_returns_filled_status(self):
        b = _broker()
        ticket = b.submit(_order())
        fill = b.get_fill(ticket)
        assert fill.status == OrderStatus.FILLED

    def test_get_fill_unknown_order(self):
        b = _broker()
        fill = b.get_fill("nonexistent_ticket")
        assert fill.status == OrderStatus.REJECTED
        assert "not_found" in fill.error

    def test_cancel_pending_order(self):
        """Only pending orders can be cancelled — filled ones cannot."""
        b = _broker()
        # Market orders fill immediately, so cancel should return False
        ticket = b.submit(_order())
        cancelled = b.cancel(ticket)
        assert not cancelled  # already filled

    def test_slippage_adds_to_long_fill(self):
        """With slippage enabled, long fill > ask."""
        b = _broker(slippage=1.0, seed=1)
        b.update_price("XAUUSD", bid=1999.70, ask=2000.30)
        order = _order(direction=Direction.LONG)
        b.submit(order)
        assert order.fill_price > 2000.30

    def test_commission_deducted_on_fill(self):
        """Commission reduces balance on fill."""
        b = _broker(equity=10_000, commission=7.0, slippage=0.0)
        initial_balance = b._balance
        b.submit(_order(lot_size=1.0))
        assert b._balance < initial_balance  # commission deducted


# ===========================================================================
# PaperBroker — position management
# ===========================================================================

class TestPaperBrokerPositions:
    def test_long_opens_position(self):
        b = _broker()
        b.submit(_order(direction=Direction.LONG))
        positions = b.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "XAUUSD"
        assert positions[0].direction == Direction.LONG

    def test_short_opens_position(self):
        b = _broker()
        b.submit(_order(direction=Direction.SHORT))
        positions = b.get_positions()
        assert len(positions) == 1
        assert positions[0].direction == Direction.SHORT

    def test_no_positions_initially(self):
        b = _broker()
        assert b.get_positions() == []

    def test_close_position_removes_it(self):
        b = _broker()
        b.submit(_order())
        assert len(b.get_positions()) == 1
        b.close_position("XAUUSD")
        assert len(b.get_positions()) == 0

    def test_close_nonexistent_position_rejected(self):
        b = _broker()
        fill = b.close_position("EURUSD")
        assert fill.status == OrderStatus.REJECTED

    def test_multiple_symbols_tracked_separately(self):
        b = _broker()
        b.submit(_order(symbol="XAUUSD", direction=Direction.LONG))
        b.submit(_order(symbol="EURUSD", direction=Direction.SHORT))
        positions = b.get_positions()
        assert len(positions) == 2
        symbols = {p.symbol for p in positions}
        assert symbols == {"XAUUSD", "EURUSD"}

    def test_position_has_correct_lot_size(self):
        b = _broker()
        b.submit(_order(lot_size=0.25))
        pos = b.get_positions()[0]
        assert pos.lot_size == pytest.approx(0.25)

    def test_position_has_stop_and_tp(self):
        b = _broker()
        b.submit(_order(stop_loss=1985.0, take_profit=2030.0))
        pos = b.get_positions()[0]
        assert pos.stop_loss == pytest.approx(1985.0)
        assert pos.take_profit == pytest.approx(2030.0)


# ===========================================================================
# PaperBroker — P&L and equity
# ===========================================================================

class TestPaperBrokerPnL:
    def test_long_profitable_close(self):
        """Open long at 2000.30 (ask), close at 2010 bid → profit."""
        b = _broker(equity=10_000, slippage=0.0, commission=0.0)
        b.update_price("XAUUSD", bid=2000.00, ask=2000.30)
        b.submit(_order(direction=Direction.LONG, lot_size=0.1))

        # Price moves up
        b.update_price("XAUUSD", bid=2010.00, ask=2010.30)
        b.close_position("XAUUSD")

        assert b.realized_pnl > 0

    def test_long_losing_close(self):
        """Open long, price drops → loss."""
        b = _broker(equity=10_000, slippage=0.0, commission=0.0)
        b.update_price("XAUUSD", bid=2000.00, ask=2000.30)
        b.submit(_order(direction=Direction.LONG, lot_size=0.1))

        b.update_price("XAUUSD", bid=1990.00, ask=1990.30)
        b.close_position("XAUUSD")

        assert b.realized_pnl < 0

    def test_equity_updates_with_unrealized_pnl(self):
        """Equity changes as price moves against open position."""
        b = _broker(equity=10_000, slippage=0.0, commission=0.0)
        b.update_price("XAUUSD", bid=2000.00, ask=2000.30)
        b.submit(_order(direction=Direction.LONG, lot_size=0.1))

        initial_equity = b.equity
        # Price drops → unrealized loss → equity decreases
        b.update_price("XAUUSD", bid=1990.00, ask=1990.30)
        assert b.equity < initial_equity

    def test_trade_log_recorded(self):
        b = _broker()
        b.submit(_order())
        b.close_position("XAUUSD")
        log = b.trade_log
        assert len(log) == 1
        assert "pnl" in log[0]
        assert log[0]["symbol"] == "XAUUSD"

    def test_performance_summary_empty(self):
        b = _broker()
        summary = b.performance_summary()
        assert summary["trades"] == 0

    def test_performance_summary_after_trades(self):
        b = _broker(slippage=0.0, commission=0.0)
        # Open and close a position
        b.update_price("XAUUSD", bid=2000.00, ask=2000.30)
        b.submit(_order(lot_size=0.1))
        b.update_price("XAUUSD", bid=2010.00, ask=2010.30)
        b.close_position("XAUUSD")

        summary = b.performance_summary()
        assert summary["trades"] == 1
        assert "win_rate" in summary
        assert "profit_factor" in summary
        assert "total_pnl" in summary

    def test_win_rate_calculation(self):
        """Two trades: one win, one loss → win_rate = 0.5."""
        b = _broker(slippage=0.0, commission=0.0)

        # Trade 1: WIN — long, price goes up
        b.update_price("XAUUSD", bid=2000.00, ask=2000.00)
        b.submit(_order(lot_size=0.1))
        b.update_price("XAUUSD", bid=2010.00, ask=2010.00)
        b.close_position("XAUUSD")

        # Trade 2: LOSS — long, price goes down
        b.update_price("XAUUSD", bid=2010.00, ask=2010.00)
        b.submit(_order(lot_size=0.1))
        b.update_price("XAUUSD", bid=2000.00, ask=2000.00)
        b.close_position("XAUUSD")

        summary = b.performance_summary()
        assert summary["trades"] == 2
        assert summary["win_rate"] == pytest.approx(0.5)


# ===========================================================================
# PaperBroker — exit order routing
# ===========================================================================

class TestPaperBrokerExitOrders:
    def test_exit_signal_closes_position(self):
        """Submitting a FLAT order should close an existing position."""
        b = _broker()
        # Open position
        b.submit(_order(direction=Direction.LONG))
        assert len(b.get_positions()) == 1

        # Submit exit order
        exit_order = Order(
            signal=_signal(direction=Direction.FLAT),
            lot_size=0.1,
            order_type=OrderType.MARKET,
        )
        b.submit(exit_order)
        assert len(b.get_positions()) == 0