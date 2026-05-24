"""
execution/broker_base.py
------------------------
Abstract broker interface.

Every broker adapter (MT5, paper, future exchanges) implements this.
The rest of the system only speaks to BrokerBase — never to a concrete
broker directly. Swapping brokers requires zero changes to strategy,
risk, or backtesting code.

Lifecycle of an order through this layer:
  RiskEngine → Order(PENDING)
  broker.submit(order) → Order(SUBMITTED), order_id assigned
  broker.get_fill(order_id) → Order(FILLED), fill_price/fill_time set
  OR
  broker.cancel(order_id) → Order(CANCELLED)

Position reconciliation:
  broker.get_positions() returns the broker's current view of open
  positions. On startup and periodically, the live trader reconciles
  this against its internal AccountState to detect desync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from risk.order import Order, OrderStatus
from strategies.base_strategy import Direction


# ---------------------------------------------------------------------------
# Position as reported by the broker
# ---------------------------------------------------------------------------

@dataclass
class BrokerPosition:
    """
    One open position as reported by the broker's API.
    This is the ground truth — used for reconciliation.
    """
    symbol: str
    direction: Direction
    lot_size: float
    open_price: float
    current_price: float
    unrealized_pnl: float
    open_time: datetime
    ticket: str = ""              # broker-assigned position ID
    stop_loss: float | None = None
    take_profit: float | None = None
    swap: float = 0.0             # overnight financing cost
    commission: float = 0.0


@dataclass
class BrokerAccountInfo:
    """Account snapshot as reported by the broker."""
    equity: float
    balance: float
    margin_used: float
    margin_free: float
    profit: float
    currency: str = "USD"
    leverage: int = 100
    broker_name: str = ""


@dataclass
class FillResult:
    """Result of a fill query for a submitted order."""
    order_id: str
    status: OrderStatus
    fill_price: float | None = None
    fill_time: datetime | None = None
    slippage: float = 0.0         # fill_price - requested_price
    commission: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Abstract broker
# ---------------------------------------------------------------------------

class BrokerBase(ABC):
    """
    Abstract interface all broker adapters must implement.

    Thread safety: implementations must be safe to call from an
    asyncio executor (MT5) or directly (paper broker).

    Error handling: methods raise BrokerError on unrecoverable failures.
    Transient errors (network timeout) should be retried internally
    before raising.
    """

    def __init__(self, broker_id: str) -> None:
        self.broker_id = broker_id
        self._connected = False
        self._submitted_orders: dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish connection. Raise BrokerError on failure."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection gracefully."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    @abstractmethod
    def get_account_info(self) -> BrokerAccountInfo:
        """Return current account equity, balance, margin."""
        ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Return all currently open positions."""
        ...

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(self, order: Order) -> str:
        """
        Submit an order to the broker.

        Returns the broker-assigned order_id (ticket number).
        Sets order.status = SUBMITTED and order.order_id = returned id.
        Raises BrokerError if submission fails.
        """
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        Returns True if cancelled, False if already filled/not found.
        """
        ...

    @abstractmethod
    def get_fill(self, order_id: str) -> FillResult:
        """
        Query fill status of a submitted order.
        Used for polling-based fill confirmation.
        """
        ...

    @abstractmethod
    def close_position(self, symbol: str, lot_size: float | None = None) -> FillResult:
        """
        Close an open position for symbol.
        lot_size=None closes the entire position.
        Returns fill result of the closing trade.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(tz=timezone.utc)

    def track(self, order: Order) -> None:
        """Register a submitted order for internal tracking."""
        if order.order_id:
            self._submitted_orders[order.order_id] = order

    def get_tracked(self, order_id: str) -> Order | None:
        return self._submitted_orders.get(order_id)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BrokerBase":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Broker error
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    """Raised on unrecoverable broker API errors."""
    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code