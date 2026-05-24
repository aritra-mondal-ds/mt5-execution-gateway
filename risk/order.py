"""
risk/order.py
-------------
Order dataclass — the atomic unit passed from RiskEngine to Execution.

A Signal becomes an Order only after passing all risk checks.
The Order carries the approved lot size alongside all execution parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from strategies.base_strategy import Direction, Signal


class OrderType(Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


class OrderStatus(Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


@dataclass
class Order:
    """
    A risk-approved trading order ready for execution.

    Fields set by RiskEngine:
        signal          : originating Signal
        lot_size        : approved position size in lots
        order_type      : MARKET / LIMIT / STOP
        stop_loss       : absolute SL price
        take_profit     : absolute TP price
        risk_fraction   : fraction of equity risked (for audit trail)
        expected_cost   : estimated spread + commission in account currency

    Fields set by Execution layer after submission:
        order_id        : broker-assigned order ID
        status          : tracks order lifecycle
        fill_price      : actual execution price
        fill_time       : UTC datetime of fill
    """
    signal: Signal
    lot_size: float
    order_type: OrderType = OrderType.MARKET
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_fraction: float = 0.0
    expected_cost: float = 0.0

    # Set post-submission by execution layer
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_time: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    @property
    def strategy_id(self) -> str:
        return self.signal.strategy_id

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_entry(self) -> bool:
        return self.signal.is_entry

    @property
    def is_exit(self) -> bool:
        return self.signal.is_exit

    def __repr__(self) -> str:
        return (
            f"Order({self.symbol} | {self.direction.value.upper()} | "
            f"lots={self.lot_size:.2f} | SL={self.stop_loss} | "
            f"TP={self.take_profit} | status={self.status.value})"
        )