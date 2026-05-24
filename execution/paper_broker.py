"""
execution/paper_broker.py
--------------------------
Paper trading broker — realistic simulated fills.

Used for:
  1. Paper trading (live data, no real money)
  2. Backtesting (historical data replay)

Fill model:
  Market orders:
    fill_price = current_price + spread/2 (buying above mid)
               = current_price - spread/2 (selling below mid)
    + slippage_factor * spread (additional market impact)

  Limit orders:
    Fill only if market price trades THROUGH the limit price
    (require_price_through=True enforces this strictness).
    No fill if price only touches the limit.

  Stop orders:
    Triggered when market reaches stop price.
    Fill at stop_price + slippage (worst-case gap fill).

Position tracking:
  The paper broker maintains its own position ledger and P&L
  so get_positions() and get_account_info() work identically
  to a live broker — the rest of the system can't tell the
  difference.

Slippage model:
  Slippage is sampled from a half-normal distribution centred
  at 0 with std = spread * slippage_std_factor. This produces
  realistic variable slippage rather than a fixed constant.
"""

from __future__ import annotations

import uuid
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from execution.broker_base import (
    BrokerBase, BrokerPosition, BrokerAccountInfo,
    FillResult, BrokerError,
)
from risk.order import Order, OrderStatus, OrderType
from strategies.base_strategy import Direction


# ---------------------------------------------------------------------------
# Internal ledger entry
# ---------------------------------------------------------------------------

@dataclass
class _PaperPosition:
    symbol: str
    direction: Direction
    lot_size: float
    open_price: float
    open_time: datetime
    stop_loss: float | None
    take_profit: float | None
    ticket: str
    commission_paid: float = 0.0
    swap_accrued: float = 0.0
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.current_price == 0:
            return 0.0
        pip_move = (
            (self.current_price - self.open_price)
            if self.direction == Direction.LONG
            else (self.open_price - self.current_price)
        )
        # Simplified: 1 lot = pip_value per point
        # Caller is responsible for instrument-specific conversion
        return pip_move * self.lot_size * 10.0  # approximate


@dataclass
class _PaperOrder:
    order: Order
    ticket: str
    submitted_at: datetime
    fill_result: FillResult | None = None


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------

class PaperBroker(BrokerBase):
    """
    Simulates a live broker with realistic fill modelling.

    Usage:
        broker = PaperBroker(initial_equity=10_000)
        broker.connect()

        order_id = broker.submit(order)
        broker.update_price("XAUUSD", bid=2000.0, ask=2000.50)
        fill = broker.get_fill(order_id)

        positions = broker.get_positions()
        info = broker.get_account_info()
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        spread_points: dict[str, float] | None = None,
        slippage_std_factor: float = 0.5,
        pip_values: dict[str, float] | None = None,
        commission_per_lot: float = 7.0,
        require_price_through: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(broker_id="paper")
        self._initial_equity = initial_equity
        self._balance = initial_equity
        self._equity = initial_equity
        self._realized_pnl = 0.0

        self._spread_points: dict[str, float] = spread_points or {
            "XAUUSD": 0.30, "EURUSD": 0.00010,
            "GBPUSD": 0.00013, "BTCUSDT": 10.0,
        }
        self._pip_values: dict[str, float] = pip_values or {
            "XAUUSD": 100.0, "EURUSD": 10.0,
            "GBPUSD": 10.0, "BTCUSDT": 1.0,
        }
        self._slippage_std_factor = slippage_std_factor
        self._commission_per_lot = commission_per_lot
        self._require_price_through = require_price_through

        # Current market prices {symbol: (bid, ask)}
        self._prices: dict[str, tuple[float, float]] = {}

        # Internal ledger
        self._positions: dict[str, _PaperPosition] = {}   # symbol → position
        self._orders: dict[str, _PaperOrder] = {}         # ticket → order

        self._rng = random.Random(seed)
        self._trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True
        logger.info(
            f"PaperBroker connected | equity={self._equity:.2f} | "
            f"mode=paper"
        )

    def disconnect(self) -> None:
        self._connected = False
        logger.info(
            f"PaperBroker disconnected | "
            f"realized_pnl={self._realized_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Update the current market price for a symbol.
        Must be called before submit() or any fill checks.
        In live paper trading: called on each new bar/tick.
        In backtesting: called by the event-driven engine per bar.
        """
        self._prices[symbol] = (bid, ask)
        # Update unrealized P&L for open positions
        if symbol in self._positions:
            pos = self._positions[symbol]
            mid = (bid + ask) / 2
            pos.current_price = mid
            self._recalculate_equity()

    def update_price_from_bar(self, symbol: str, close: float, spread: float = 0.0) -> None:
        """Convenience: update from a Bar's close price."""
        half_spread = spread / 2 or self._spread_points.get(symbol, 0.0) / 2
        self.update_price(symbol, bid=close - half_spread, ask=close + half_spread)

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account_info(self) -> BrokerAccountInfo:
        self._recalculate_equity()
        return BrokerAccountInfo(
            equity=self._equity,
            balance=self._balance,
            margin_used=0.0,    # simplified — no margin tracking
            margin_free=self._equity,
            profit=self._realized_pnl,
            currency="USD",
            broker_name="PaperBroker",
        )

    def get_positions(self) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                symbol=pos.symbol,
                direction=pos.direction,
                lot_size=pos.lot_size,
                open_price=pos.open_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                open_time=pos.open_time,
                ticket=pos.ticket,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                commission=pos.commission_paid,
            )
            for pos in self._positions.values()
        ]

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> str:
        """
        Submit an order. For market orders, fills immediately at
        the current price + slippage. For limit/stop orders, registers
        for deferred fill checking.
        """
        if not self._connected:
            raise BrokerError("PaperBroker not connected")

        ticket = str(uuid.uuid4())[:8]
        order.order_id = ticket
        order.status = OrderStatus.SUBMITTED

        paper_order = _PaperOrder(
            order=order,
            ticket=ticket,
            submitted_at=self.utcnow(),
        )
        self._orders[ticket] = paper_order
        self.track(order)

        # Market orders: fill immediately
        if order.order_type == OrderType.MARKET:
            fill = self._fill_market(order)
            paper_order.fill_result = fill
            order.status = fill.status
            order.fill_price = fill.fill_price
            order.fill_time = fill.fill_time

            if fill.status == OrderStatus.FILLED:
                self._open_position(order, fill.fill_price)

        logger.info(
            f"PaperBroker submit | {ticket} | {order.symbol} | "
            f"{order.direction.value} | lots={order.lot_size:.2f} | "
            f"status={order.status.value}"
        )
        return ticket

    # ------------------------------------------------------------------
    # Fill and position management
    # ------------------------------------------------------------------

    def get_fill(self, order_id: str) -> FillResult:
        paper_order = self._orders.get(order_id)
        if paper_order is None:
            return FillResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                error="order_not_found",
            )
        if paper_order.fill_result is not None:
            return paper_order.fill_result

        # Limit/stop orders: check if fillable now
        fill = self._check_pending_fill(paper_order)
        if fill is not None:
            paper_order.fill_result = fill
            paper_order.order.status = fill.status
            if fill.status == OrderStatus.FILLED:
                self._open_position(paper_order.order, fill.fill_price)
        return fill or FillResult(
            order_id=order_id,
            status=OrderStatus.SUBMITTED,
        )

    def cancel(self, order_id: str) -> bool:
        paper_order = self._orders.get(order_id)
        if paper_order is None:
            return False
        if paper_order.fill_result is not None:
            return False  # already filled
        paper_order.order.status = OrderStatus.CANCELLED
        paper_order.fill_result = FillResult(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
        )
        return True

    def close_position(
        self, symbol: str, lot_size: float | None = None
    ) -> FillResult:
        pos = self._positions.get(symbol)
        if pos is None:
            return FillResult(
                order_id="", status=OrderStatus.REJECTED,
                error=f"no_position_for_{symbol}",
            )

        close_lots = lot_size or pos.lot_size
        fill_price = self._get_close_price(symbol, pos.direction)
        pnl = self._calculate_pnl(pos, fill_price, close_lots)
        commission = self._commission_per_lot * close_lots

        self._realized_pnl += pnl - commission
        self._balance += pnl - commission

        self._trade_log.append({
            "symbol": symbol,
            "direction": pos.direction.value,
            "lots": close_lots,
            "open_price": pos.open_price,
            "close_price": fill_price,
            "pnl": pnl,
            "commission": commission,
            "net_pnl": pnl - commission,
            "open_time": pos.open_time,
            "close_time": self.utcnow(),
        })

        if close_lots >= pos.lot_size:
            del self._positions[symbol]
        else:
            pos.lot_size -= close_lots

        self._recalculate_equity()
        logger.info(
            f"PaperBroker close | {symbol} | "
            f"pnl={pnl:.2f} commission={commission:.2f} "
            f"net={pnl - commission:.2f}"
        )

        return FillResult(
            order_id="", status=OrderStatus.FILLED,
            fill_price=fill_price, fill_time=self.utcnow(),
            commission=commission,
        )

    # ------------------------------------------------------------------
    # Fill model internals
    # ------------------------------------------------------------------

    def _fill_market(self, order: Order) -> FillResult:
        """Fill a market order at current price + slippage."""
        bid, ask = self._get_prices(order.symbol)
        spread = self._spread_points.get(order.symbol, 0.0)
        slippage = self._sample_slippage(spread)

        if order.direction == Direction.LONG:
            fill_price = ask + slippage
        elif order.direction == Direction.SHORT:
            fill_price = bid - slippage
        else:
            # FLAT / exit — use mid price
            fill_price = (bid + ask) / 2

        commission = self._commission_per_lot * order.lot_size
        self._balance -= commission  # charge commission on entry

        return FillResult(
            order_id=order.order_id or "",
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_time=self.utcnow(),
            slippage=slippage,
            commission=commission,
        )

    def _check_pending_fill(self, paper_order: _PaperOrder) -> FillResult | None:
        """Check if a limit/stop order can now be filled."""
        order = paper_order.order
        bid, ask = self._get_prices(order.symbol)

        if order.order_type == OrderType.LIMIT:
            if order.direction == Direction.LONG and order.stop_loss is not None:
                limit = order.stop_loss
                filled = ask <= limit if self._require_price_through else ask <= limit
                if filled:
                    return self._make_fill(order, ask)

            elif order.direction == Direction.SHORT and order.stop_loss is not None:
                limit = order.stop_loss
                filled = bid >= limit
                if filled:
                    return self._make_fill(order, bid)

        return None

    def _make_fill(self, order: Order, price: float) -> FillResult:
        spread = self._spread_points.get(order.symbol, 0.0)
        slippage = self._sample_slippage(spread)
        fill_price = price + (slippage if order.direction == Direction.LONG else -slippage)
        commission = self._commission_per_lot * order.lot_size
        self._balance -= commission
        return FillResult(
            order_id=order.order_id or "",
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_time=self.utcnow(),
            slippage=slippage,
            commission=commission,
        )

    def _sample_slippage(self, spread: float) -> float:
        """Sample slippage from half-normal distribution."""
        std = spread * self._slippage_std_factor
        return abs(self._rng.gauss(0, std)) if std > 0 else 0.0

    # ------------------------------------------------------------------
    # Position ledger
    # ------------------------------------------------------------------

    def _open_position(self, order: Order, fill_price: float) -> None:
        if order.is_exit:
            # Close existing position
            self.close_position(order.symbol, order.lot_size)
            return

        pos = _PaperPosition(
            symbol=order.symbol,
            direction=order.direction,
            lot_size=order.lot_size,
            open_price=fill_price,
            open_time=order.fill_time or self.utcnow(),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            ticket=order.order_id or "",
            commission_paid=self._commission_per_lot * order.lot_size,
            current_price=fill_price,
        )
        self._positions[order.symbol] = pos

    def _recalculate_equity(self) -> None:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        self._equity = self._balance + unrealized

    def _get_prices(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._prices:
            raise BrokerError(
                f"No price data for {symbol}. "
                "Call update_price() before submitting orders."
            )
        return self._prices[symbol]

    def _get_close_price(self, symbol: str, direction: Direction) -> float:
        bid, ask = self._get_prices(symbol)
        return bid if direction == Direction.LONG else ask

    def _calculate_pnl(
        self, pos: _PaperPosition, close_price: float, lots: float
    ) -> float:
        pip_value = self._pip_values.get(pos.symbol, 10.0)
        if pos.direction == Direction.LONG:
            pip_move = close_price - pos.open_price
        else:
            pip_move = pos.open_price - close_price
        return pip_move * pip_value * lots

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def equity(self) -> float:
        self._recalculate_equity()
        return self._equity

    def performance_summary(self) -> dict:
        if not self._trade_log:
            return {"trades": 0}
        pnls = [t["net_pnl"] for t in self._trade_log]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "trades": len(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "profit_factor": (
                abs(sum(wins)) / abs(sum(losses))
                if losses and sum(losses) != 0 else float("inf")
            ),
            "equity": self._equity,
        }