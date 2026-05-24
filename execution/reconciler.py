"""
execution/reconciler.py
------------------------
Position reconciliation between MT5 broker and system internal state.

Problem this solves:
  The system's RiskEngine tracks positions internally. If the system
  restarts, or if trades were opened manually in MT5, the internal
  state is out of sync with the broker's actual positions.
  Without reconciliation the risk engine would:
    - Open duplicate positions on symbols already held
    - Miscalculate total exposure
    - Miss stop-loss hits on manually opened trades

Reconciliation does two things:
  1. STARTUP sync: reads all open MT5 positions and injects them into
     the system's AccountState so risk checks are accurate immediately.
  2. PERIODIC sync: runs every N seconds to catch manual MT5 trades,
     partial fills, and position size changes (e.g. broker margin calls).

Desync detection:
  Compares system's internal position list vs broker's position list.
  Reports:
    - Positions in broker but not in system (phantom positions)
    - Positions in system but not in broker (ghost positions)
    - Size mismatches (partial close happened outside system)

Design: reconciler is read-only with respect to MT5. It never submits
or cancels orders — it only reads and updates internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from execution.broker_base import BrokerBase, BrokerPosition
from risk.engine import AccountState, PositionInfo
from strategies.base_strategy import Direction


# ---------------------------------------------------------------------------
# Reconciliation report
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationReport:
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    broker_positions: list[BrokerPosition] = field(default_factory=list)
    system_positions: list[PositionInfo] = field(default_factory=list)

    # Discrepancies found
    phantom: list[str] = field(default_factory=list)   # in broker, not system
    ghost: list[str] = field(default_factory=list)     # in system, not broker
    size_mismatch: list[str] = field(default_factory=list)

    equity: float = 0.0
    balance: float = 0.0

    @property
    def is_clean(self) -> bool:
        return not self.phantom and not self.ghost and not self.size_mismatch

    @property
    def total_open(self) -> int:
        return len(self.broker_positions)

    def log(self) -> None:
        status = "CLEAN" if self.is_clean else "DESYNCED"
        log_fn = logger.info if not self.is_clean else logger.debug
        log_fn(
            f"Reconciliation [{status}] | "
            f"broker_positions={len(self.broker_positions)} | "
            f"equity={self.equity:.2f} | balance={self.balance:.2f}"
        )
        if self.phantom:
            for sym in self.phantom:
                logger.warning(
                    f"  PHANTOM: {sym} open in MT5 but unknown to system — "
                    "injecting into AccountState"
                )
        if self.ghost:
            for sym in self.ghost:
                logger.warning(
                    f"  GHOST: {sym} in system state but not in MT5 — "
                    "removing from AccountState"
                )
        if self.size_mismatch:
            for sym in self.size_mismatch:
                logger.warning(
                    f"  SIZE MISMATCH: {sym} lot size differs between "
                    "system and broker — using broker value"
                )


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """
    Syncs MT5 broker positions into the system's AccountState.

    Usage:
        reconciler = PositionReconciler(broker)

        # On startup — blocking, must complete before trading starts
        account_state, report = reconciler.reconcile(system_positions=[])
        report.log()

        # In trading loop — call periodically
        account_state, report = reconciler.reconcile(system_positions)
        if not report.is_clean:
            # Update internal state with corrected account_state
    """

    def __init__(self, broker: BrokerBase) -> None:
        self._broker = broker

    def reconcile(
        self,
        system_positions: list[PositionInfo],
    ) -> tuple[AccountState, ReconciliationReport]:
        """
        Read broker state and reconcile against system's internal positions.

        Returns:
            (AccountState, ReconciliationReport)

            AccountState is always built from broker truth — it is the
            authoritative source. The system's internal state should be
            updated to match it.
        """
        # 1. Get broker ground truth
        try:
            account_info = self._broker.get_account_info()
            broker_positions = self._broker.get_positions()
        except Exception as e:
            logger.error(f"Reconciler: failed to read broker state: {e}")
            # Return last known system state as fallback
            return self._fallback_state(system_positions), ReconciliationReport()

        report = ReconciliationReport(
            broker_positions=broker_positions,
            system_positions=system_positions,
            equity=account_info.equity,
            balance=account_info.balance,
        )

        # 2. Build lookup maps
        broker_map: dict[str, BrokerPosition] = {
            p.symbol: p for p in broker_positions
        }
        system_map: dict[str, PositionInfo] = {
            p.symbol: p for p in system_positions
        }

        # 3. Detect discrepancies
        broker_symbols = set(broker_map.keys())
        system_symbols = set(system_map.keys())

        report.phantom = list(broker_symbols - system_symbols)
        report.ghost   = list(system_symbols - broker_symbols)
        report.size_mismatch = [
            sym for sym in broker_symbols & system_symbols
            if abs(broker_map[sym].lot_size - system_map[sym].lot_size) > 0.001
        ]

        # 4. Build corrected PositionInfo list from broker truth
        corrected_positions = [
            self._broker_to_position_info(p)
            for p in broker_positions
        ]

        # 5. Build authoritative AccountState
        account_state = AccountState(
            equity=account_info.equity,
            balance=account_info.balance,
            open_positions=corrected_positions,
            daily_start_equity=account_info.equity,   # updated by drawdown breaker
            peak_equity=account_info.equity,
        )

        report.log()
        return account_state, report

    def get_equity(self) -> float:
        """Fast equity read without full reconciliation."""
        try:
            return self._broker.get_account_info().equity
        except Exception as e:
            logger.warning(f"Reconciler: equity read failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _broker_to_position_info(pos: BrokerPosition) -> PositionInfo:
        """Convert a BrokerPosition to the system's PositionInfo."""
        return PositionInfo(
            symbol=pos.symbol,
            direction=pos.direction,
            lot_size=pos.lot_size,
            entry_price=pos.open_price,
            strategy_id=f"mt5_ticket_{pos.ticket}",  # marks as externally opened
            open_time=pos.open_time,
        )

    @staticmethod
    def _fallback_state(system_positions: list[PositionInfo]) -> AccountState:
        """Return minimal account state from system knowledge when broker unreachable."""
        return AccountState(
            equity=0.0,
            balance=0.0,
            open_positions=system_positions,
        )