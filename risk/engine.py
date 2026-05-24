"""
risk/engine.py
--------------
The Risk Engine — the single hard gate between strategy signals and orders.

Every Signal from every strategy must pass through evaluate().
No strategy, no external code, no shortcut bypasses this.

Evaluation pipeline (in order):
  1. Drawdown circuit breaker    → halt check
  2. Duplicate position check    → no re-entry while already positioned
  3. Total exposure check        → account-wide position limit
  4. Position sizing             → calculate lot size
  5. Cost filter                 → edge must exceed transaction cost
  6. Stop loss validation        → SL must exist and be sensible
  7. Order construction          → assemble approved Order

Each check returns a RejectionReason on failure.
If all pass, an Order is returned to the execution layer.

AccountState:
  The engine is stateless regarding P&L — it receives an AccountState
  snapshot on every call. The caller (live trader / backtester) is
  responsible for keeping AccountState current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from strategies.base_strategy import Direction, Signal
from risk.order import Order, OrderType
from risk.position_sizer import PositionSizer, SizingResult
from risk.drawdown import DrawdownBreaker
from risk.cost_filter import CostFilter


# ---------------------------------------------------------------------------
# Account state snapshot
# ---------------------------------------------------------------------------

@dataclass
class PositionInfo:
    """One open position in the account."""
    symbol: str
    direction: Direction
    lot_size: float
    entry_price: float
    strategy_id: str
    open_time: datetime


@dataclass
class AccountState:
    """
    Snapshot of account state at the time of signal evaluation.
    Provided by the live trader / backtester on each call.
    """
    equity: float
    balance: float
    open_positions: list[PositionInfo] = field(default_factory=list)
    daily_start_equity: float = 0.0
    peak_equity: float = 0.0

    @property
    def total_exposure_lots(self) -> float:
        return sum(p.lot_size for p in self.open_positions)

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.open_positions)

    def position_for(self, symbol: str) -> PositionInfo | None:
        for p in self.open_positions:
            if p.symbol == symbol:
                return p
        return None


# ---------------------------------------------------------------------------
# Rejection tracking
# ---------------------------------------------------------------------------

class RejectionReason(Enum):
    DRAWDOWN_HALT        = "drawdown_halt"
    ALREADY_POSITIONED   = "already_positioned"
    EXPOSURE_LIMIT       = "exposure_limit"
    SIZING_FAILED        = "sizing_failed"
    COST_FILTER          = "cost_filter"
    INVALID_STOP         = "invalid_stop"
    ZERO_LOT_SIZE        = "zero_lot_size"


@dataclass
class EvaluationResult:
    """Full audit trail of a signal evaluation."""
    signal: Signal
    approved: bool
    order: Order | None = None
    rejection_reason: RejectionReason | None = None
    rejection_detail: str = ""
    sizing: SizingResult | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def __repr__(self) -> str:
        if self.approved:
            return f"EvaluationResult(APPROVED | {self.order})"
        return (
            f"EvaluationResult(REJECTED | "
            f"{self.rejection_reason.value if self.rejection_reason else '?'} | "
            f"{self.rejection_detail})"
        )


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Evaluates signals and produces approved Orders or rejections.

    Usage:
        engine = RiskEngine(config)
        engine.drawdown_breaker.initialize(equity=10_000)

        result = engine.evaluate(signal, account_state)
        if result.approved:
            execution_layer.submit(result.order)
        else:
            logger.info(f"Signal rejected: {result.rejection_detail}")
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.01,
        max_total_exposure: float = 0.10,
        max_positions: int = 5,
        atr_multiplier: float = 1.5,
        tp_rr_ratio: float = 2.0,
        max_daily_drawdown: float = 0.03,
        max_peak_drawdown: float = 0.08,
        min_edge_cost_ratio: float = 1.5,
        commission_per_lot: float = 7.0,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        kelly_fraction: float = 0.0,
    ) -> None:
        self._max_total_exposure = max_total_exposure
        self._max_positions = max_positions
        self._tp_rr_ratio = tp_rr_ratio

        self.drawdown_breaker = DrawdownBreaker(
            max_daily_drawdown=max_daily_drawdown,
            max_peak_drawdown=max_peak_drawdown,
        )
        self._sizer = PositionSizer(
            max_risk_per_trade=max_risk_per_trade,
            atr_multiplier=atr_multiplier,
            min_lot=min_lot,
            max_lot=max_lot,
            kelly_fraction=kelly_fraction,
        )
        self._cost_filter = CostFilter(
            min_edge_cost_ratio=min_edge_cost_ratio,
            commission_per_lot=commission_per_lot,
        )

        # Audit log: last N evaluation results
        self._audit_log: list[EvaluationResult] = []
        self._max_audit_log = 500

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: Signal,
        account: AccountState,
    ) -> EvaluationResult:
        """
        Evaluate a signal against all risk checks.

        Exit signals (Direction.FLAT) bypass sizing and cost checks —
        we always allow closing a position.
        """
        # EXIT signals: minimal checks, always approved if position exists
        if signal.is_exit:
            return self._approve_exit(signal, account)

        # ENTRY signals: full pipeline
        result = self._evaluate_entry(signal, account)

        # Audit
        self._audit_log.append(result)
        if len(self._audit_log) > self._max_audit_log:
            self._audit_log.pop(0)

        if result.approved:
            logger.info(f"RiskEngine APPROVED | {result.order}")
        else:
            logger.info(
                f"RiskEngine REJECTED | {signal.strategy_id} | "
                f"{signal.symbol} | {result.rejection_reason.value} | "
                f"{result.rejection_detail}"
            )

        return result

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def _approve_exit(
        self, signal: Signal, account: AccountState
    ) -> EvaluationResult:
        """
        Exit signals bypass sizing — we close at market with existing lot size.
        """
        pos = account.position_for(signal.symbol)
        lot_size = pos.lot_size if pos else self._sizer.min_lot

        order = Order(
            signal=signal,
            lot_size=lot_size,
            order_type=OrderType.MARKET,
            stop_loss=None,
            take_profit=None,
        )
        return EvaluationResult(signal=signal, approved=True, order=order)

    # ------------------------------------------------------------------
    # Entry pipeline
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self, signal: Signal, account: AccountState
    ) -> EvaluationResult:
        def reject(reason: RejectionReason, detail: str = "") -> EvaluationResult:
            return EvaluationResult(
                signal=signal,
                approved=False,
                rejection_reason=reason,
                rejection_detail=detail,
            )

        # 1. Drawdown circuit breaker
        if self.drawdown_breaker.is_halted:
            return reject(
                RejectionReason.DRAWDOWN_HALT,
                self.drawdown_breaker.state.halt_reason
                if self.drawdown_breaker.state else "not_initialized",
            )

        # 2. Already have a SYSTEM position in this symbol?
        # External positions (opened manually in MT5, strategy_id starts with
        # "mt5_ticket_") do NOT block the system from trading the same symbol.
        system_positions = [
            p for p in account.open_positions
            if not p.strategy_id.startswith("mt5_ticket_")
        ]
        system_symbols = {p.symbol for p in system_positions}
        if signal.symbol in system_symbols:
            return reject(
                RejectionReason.ALREADY_POSITIONED,
                f"existing system position in {signal.symbol}",
            )

        # 3. Total SYSTEM position count / exposure limit
        # External positions don't count toward max_positions — we don't
        # control them. Only system-opened trades are capped.
        if len(system_positions) >= self._max_positions:
            return reject(
                RejectionReason.EXPOSURE_LIMIT,
                f"max_positions={self._max_positions} reached (system positions only)",
            )

        exposure_fraction = (
            account.total_exposure_lots /
            max(account.equity, 1.0)
        ) if account.open_positions else 0.0
        if exposure_fraction > self._max_total_exposure:
            return reject(
                RejectionReason.EXPOSURE_LIMIT,
                f"total_exposure={exposure_fraction:.2%} > {self._max_total_exposure:.2%}",
            )

        # 4. Position sizing
        atr = signal.atr_at_signal
        stop_loss = signal.stop_loss
        stop_distance = None
        close_price = None

        # Derive stop distance from signal's SL if available
        if stop_loss is not None:
            # We need close price — stored in signal.meta if set by strategy
            close_price = signal.meta.get("close")
            if close_price is not None:
                stop_distance = abs(close_price - stop_loss)

        sizing = self._sizer.calculate(
            equity=account.equity,
            atr=atr,
            symbol=signal.symbol,
            stop_distance=stop_distance,
        )

        if sizing.rejection_reason:
            return EvaluationResult(
                signal=signal, approved=False,
                rejection_reason=RejectionReason.SIZING_FAILED,
                rejection_detail=sizing.rejection_reason,
                sizing=sizing,
            )

        if sizing.lot_size <= 0:
            return EvaluationResult(
                signal=signal, approved=False,
                rejection_reason=RejectionReason.ZERO_LOT_SIZE,
                sizing=sizing,
            )

        # 5. Cost filter (entry signals only)
        rr_ratio = self._tp_rr_ratio
        cost_check = self._cost_filter.evaluate(
            symbol=signal.symbol,
            lot_size=sizing.lot_size,
            stop_distance=sizing.stop_distance,
            rr_ratio=rr_ratio,
            signal_strength=signal.strength,
        )
        if not cost_check.passed:
            return EvaluationResult(
                signal=signal, approved=False,
                rejection_reason=RejectionReason.COST_FILTER,
                rejection_detail=cost_check.rejection_reason,
                sizing=sizing,
            )

        # 6. Stop loss validation
        final_sl = signal.stop_loss
        final_tp = signal.take_profit

        # If strategy didn't set SL, compute from ATR
        if final_sl is None and atr is not None and close_price is not None:
            sl_dist = atr * self._sizer.atr_multiplier
            final_sl = (
                close_price - sl_dist
                if signal.direction == Direction.LONG
                else close_price + sl_dist
            )
            final_tp = (
                close_price + sl_dist * rr_ratio
                if signal.direction == Direction.LONG
                else close_price - sl_dist * rr_ratio
            )

        # 7. Build approved order
        order = Order(
            signal=signal,
            lot_size=sizing.lot_size,
            order_type=OrderType.MARKET,
            stop_loss=final_sl,
            take_profit=final_tp,
            risk_fraction=sizing.risk_fraction,
            expected_cost=cost_check.total_cost,
        )

        return EvaluationResult(
            signal=signal, approved=True,
            order=order, sizing=sizing,
        )

    # ------------------------------------------------------------------
    # Audit / diagnostics
    # ------------------------------------------------------------------

    @property
    def audit_log(self) -> list[EvaluationResult]:
        return list(self._audit_log)

    def rejection_stats(self) -> dict[str, int]:
        """Count rejections by reason — useful for strategy diagnostics."""
        counts: dict[str, int] = {}
        for r in self._audit_log:
            if not r.approved and r.rejection_reason:
                k = r.rejection_reason.value
                counts[k] = counts.get(k, 0) + 1
        return counts