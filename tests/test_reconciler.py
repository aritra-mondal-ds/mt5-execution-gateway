"""
tests/test_reconciler.py
-------------------------
Tests for PositionReconciler.

Run with:
    pytest tests/test_reconciler.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from execution.broker_base import BrokerPosition, BrokerAccountInfo
from execution.reconciler import PositionReconciler, ReconciliationReport
from risk.engine import PositionInfo
from strategies.base_strategy import Direction

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _broker_pos(symbol: str, lots: float = 0.1,
                direction: Direction = Direction.LONG) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol, direction=direction,
        lot_size=lots, open_price=2000.0, current_price=2005.0,
        unrealized_pnl=50.0, open_time=_NOW, ticket="12345",
    )


def _sys_pos(symbol: str, lots: float = 0.1,
             direction: Direction = Direction.LONG) -> PositionInfo:
    return PositionInfo(
        symbol=symbol, direction=direction, lot_size=lots,
        entry_price=2000.0, strategy_id="trend", open_time=_NOW,
    )


def _mock_broker(positions=None, equity=10_000.0, balance=10_000.0):
    broker = MagicMock()
    broker.get_account_info.return_value = BrokerAccountInfo(
        equity=equity, balance=balance,
        margin_used=0, margin_free=equity, profit=0,
    )
    broker.get_positions.return_value = positions or []
    return broker


class TestReconciliationReport:
    def test_clean_when_no_discrepancies(self):
        report = ReconciliationReport()
        assert report.is_clean

    def test_not_clean_with_phantom(self):
        report = ReconciliationReport(phantom=["XAUUSD"])
        assert not report.is_clean

    def test_not_clean_with_ghost(self):
        report = ReconciliationReport(ghost=["EURUSD"])
        assert not report.is_clean

    def test_not_clean_with_size_mismatch(self):
        report = ReconciliationReport(size_mismatch=["XAUUSD"])
        assert not report.is_clean

    def test_total_open_matches_broker_positions(self):
        report = ReconciliationReport(
            broker_positions=[_broker_pos("XAUUSD"), _broker_pos("EURUSD")]
        )
        assert report.total_open == 2


class TestPositionReconciler:
    def test_clean_reconcile_no_positions(self):
        broker = _mock_broker(positions=[], equity=10_000.0)
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(system_positions=[])
        assert report.is_clean
        assert len(state.open_positions) == 0
        assert state.equity == pytest.approx(10_000.0)

    def test_broker_position_injected_into_state(self):
        """MT5 has a position the system doesn't know about → phantom → inject."""
        broker = _mock_broker(positions=[_broker_pos("XAUUSD", lots=0.2)])
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(system_positions=[])

        assert "XAUUSD" in report.phantom
        assert len(state.open_positions) == 1
        assert state.open_positions[0].symbol == "XAUUSD"
        assert state.open_positions[0].lot_size == pytest.approx(0.2)

    def test_ghost_position_detected(self):
        """System thinks it has EURUSD but MT5 doesn't — ghost."""
        broker = _mock_broker(positions=[])
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(
            system_positions=[_sys_pos("EURUSD")]
        )
        assert "EURUSD" in report.ghost
        # AccountState is built from broker truth → EURUSD removed
        assert len(state.open_positions) == 0

    def test_size_mismatch_detected(self):
        """Broker has 0.5 lots but system tracks 0.1 lots."""
        broker = _mock_broker(positions=[_broker_pos("XAUUSD", lots=0.5)])
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(
            system_positions=[_sys_pos("XAUUSD", lots=0.1)]
        )
        assert "XAUUSD" in report.size_mismatch
        # State uses broker truth
        assert state.open_positions[0].lot_size == pytest.approx(0.5)

    def test_clean_when_positions_match(self):
        broker = _mock_broker(positions=[_broker_pos("XAUUSD", lots=0.1)])
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(
            system_positions=[_sys_pos("XAUUSD", lots=0.1)]
        )
        assert report.is_clean

    def test_multiple_positions_all_injected(self):
        broker = _mock_broker(positions=[
            _broker_pos("XAUUSD", lots=0.1),
            _broker_pos("EURUSD", lots=0.2),
        ])
        rec = PositionReconciler(broker)
        state, report = rec.reconcile(system_positions=[])
        assert len(state.open_positions) == 2
        symbols = {p.symbol for p in state.open_positions}
        assert symbols == {"XAUUSD", "EURUSD"}

    def test_equity_from_broker(self):
        broker = _mock_broker(equity=9_500.0)
        rec = PositionReconciler(broker)
        state, _ = rec.reconcile(system_positions=[])
        assert state.equity == pytest.approx(9_500.0)

    def test_broker_failure_returns_fallback(self):
        """If broker is unreachable, return system's last known state."""
        broker = MagicMock()
        broker.get_account_info.side_effect = ConnectionError("MT5 down")
        broker.get_positions.side_effect = ConnectionError("MT5 down")

        rec = PositionReconciler(broker)
        sys_positions = [_sys_pos("XAUUSD")]
        state, report = rec.reconcile(system_positions=sys_positions)

        # Falls back to system state, equity = 0 (unknown)
        assert state.equity == 0.0
        assert len(state.open_positions) == 1

    def test_strategy_id_marks_external_positions(self):
        """Positions opened outside the system get a ticket-based strategy_id."""
        broker = _mock_broker(positions=[_broker_pos("XAUUSD")])
        rec = PositionReconciler(broker)
        state, _ = rec.reconcile(system_positions=[])
        assert "mt5_ticket" in state.open_positions[0].strategy_id

    def test_get_equity_fast_path(self):
        broker = _mock_broker(equity=12_345.67)
        rec = PositionReconciler(broker)
        assert rec.get_equity() == pytest.approx(12_345.67)

    def test_get_equity_failure_returns_zero(self):
        broker = MagicMock()
        broker.get_account_info.side_effect = Exception("timeout")
        rec = PositionReconciler(broker)
        assert rec.get_equity() == 0.0

    def test_direction_preserved_from_broker(self):
        broker = _mock_broker(positions=[
            _broker_pos("XAUUSD", direction=Direction.SHORT)
        ])
        rec = PositionReconciler(broker)
        state, _ = rec.reconcile(system_positions=[])
        assert state.open_positions[0].direction == Direction.SHORT