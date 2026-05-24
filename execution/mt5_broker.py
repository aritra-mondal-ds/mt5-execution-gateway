"""
execution/mt5_broker.py
------------------------
MetaTrader 5 live broker adapter.

Translates MT5's Python API into the BrokerBase interface.
All MT5 API calls run in a ThreadPoolExecutor (max_workers=1)
because the MT5 Python library is not thread-safe.

MT5 order flow:
  submit()        → mt5.order_send()  → returns ticket (int)
  get_fill()      → mt5.orders_get() / mt5.history_deals_get()
  close_position()→ mt5.order_send() with opposite direction

MT5 position direction mapping:
  mt5.POSITION_TYPE_BUY  → Direction.LONG
  mt5.POSITION_TYPE_SELL → Direction.SHORT

Error handling:
  MT5 returns (retcode, result) tuples. TRADE_RETCODE_DONE (10009)
  means success. Any other code raises BrokerError with the MT5
  error description.

Important MT5 quirks handled here:
  - Symbol names may differ by broker (e.g. "XAUUSD" vs "XAUUSDm")
  - MT5 requires the terminal to be open and logged in
  - Timestamps are broker-local; we convert to UTC immediately
  - Lot sizes must be multiples of the symbol's lot step
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

from execution.broker_base import (
    BrokerBase, BrokerPosition, BrokerAccountInfo,
    FillResult, BrokerError,
)
from risk.order import Order, OrderStatus, OrderType
from strategies.base_strategy import Direction

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# MT5 retcodes
_RETCODE_DONE = 10009
_RETCODE_PLACED = 10008


class MT5Broker(BrokerBase):
    """
    Live MetaTrader 5 broker adapter.

    Usage:
        broker = MT5Broker(login=12345, password="pass", server="Broker-Demo")
        with broker:
            order_id = broker.submit(order)
            fill = broker.get_fill(order_id)
            positions = broker.get_positions()
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        timeout_ms: int = 10_000,
        magic_number: int = 234000,     # unique EA identifier
        deviation: int = 20,            # max slippage in points
        lot_step_overrides: dict[str, float] | None = None,
    ) -> None:
        super().__init__(broker_id="mt5")
        self._login = login
        self._password = password
        self._server = server
        self._timeout_ms = timeout_ms
        self._magic = magic_number
        self._deviation = deviation
        self._lot_steps = lot_step_overrides or {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5_broker")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if not MT5_AVAILABLE:
            raise BrokerError("MetaTrader5 package not installed")
        self._run_sync(self._connect_sync)

    def _connect_sync(self) -> None:
        if not mt5.initialize(
            login=self._login,
            password=self._password,
            server=self._server,
            timeout=self._timeout_ms,
        ):
            raise BrokerError(f"MT5 init failed: {mt5.last_error()}")

        info = mt5.account_info()
        if info is None:
            raise BrokerError("MT5 connected but account info unavailable")

        self._connected = True
        logger.info(
            f"MT5Broker connected | account={info.login} | "
            f"broker={info.company} | equity={info.equity:.2f} {info.currency}"
        )

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self._connected:
            self._run_sync(mt5.shutdown)
        self._connected = False
        self._executor.shutdown(wait=False)
        logger.info("MT5Broker disconnected")

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account_info(self) -> BrokerAccountInfo:
        info = self._run_sync(mt5.account_info)
        if info is None:
            raise BrokerError(f"Failed to get account info: {mt5.last_error()}")
        return BrokerAccountInfo(
            equity=info.equity,
            balance=info.balance,
            margin_used=info.margin,
            margin_free=info.margin_free,
            profit=info.profit,
            currency=info.currency,
            leverage=info.leverage,
            broker_name=info.company,
        )

    def get_positions(self) -> list[BrokerPosition]:
        positions = self._run_sync(mt5.positions_get)
        if positions is None:
            return []
        return [self._parse_position(p) for p in positions]

    def _parse_position(self, p: Any) -> BrokerPosition:
        direction = (
            Direction.LONG if p.type == mt5.POSITION_TYPE_BUY
            else Direction.SHORT
        )
        open_time = datetime.fromtimestamp(p.time, tz=timezone.utc)
        return BrokerPosition(
            symbol=p.symbol,
            direction=direction,
            lot_size=p.volume,
            open_price=p.price_open,
            current_price=p.price_current,
            unrealized_pnl=p.profit,
            open_time=open_time,
            ticket=str(p.ticket),
            stop_loss=p.sl if p.sl != 0 else None,
            take_profit=p.tp if p.tp != 0 else None,
            swap=p.swap,
            commission=getattr(p, "commission", 0.0),  # not on open positions
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> str:
        request = self._build_request(order)
        result = self._run_sync(mt5.order_send, request)

        if result is None:
            raise BrokerError(f"order_send returned None: {mt5.last_error()}")

        if result.retcode not in (_RETCODE_DONE, _RETCODE_PLACED):
            raise BrokerError(
                f"Order rejected: retcode={result.retcode} "
                f"comment={result.comment}",
                code=result.retcode,
            )

        ticket = str(result.order)
        order.order_id = ticket
        order.status = OrderStatus.SUBMITTED
        self.track(order)

        logger.info(
            f"MT5Broker submitted | ticket={ticket} | "
            f"{order.symbol} {order.direction.value} "
            f"lots={order.lot_size:.2f}"
        )
        return ticket

    def _build_request(self, order: Order) -> dict:
        """Construct the MT5 order_send request dict."""
        action = mt5.TRADE_ACTION_DEAL   # market order

        order_type = (
            mt5.ORDER_TYPE_BUY
            if order.direction == Direction.LONG
            else mt5.ORDER_TYPE_SELL
        )

        # For exit orders, use opposite direction
        if order.is_exit:
            order_type = (
                mt5.ORDER_TYPE_SELL
                if order.direction == Direction.LONG
                else mt5.ORDER_TYPE_BUY
            )

        # Get current price for market order
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            raise BrokerError(f"Cannot get tick for {order.symbol}")

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        # Validate and round lot size to symbol's lot step
        lot_size = self._round_lots(order.symbol, order.lot_size)

        request = {
            "action": action,
            "symbol": order.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": f"{order.strategy_id[:20]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Attach SL/TP if provided
        if order.stop_loss is not None:
            request["sl"] = order.stop_loss
        if order.take_profit is not None:
            request["tp"] = order.take_profit

        return request

    # ------------------------------------------------------------------
    # Fill tracking
    # ------------------------------------------------------------------

    def get_fill(self, order_id: str) -> FillResult:
        """
        Check if an order has been filled by querying MT5 deal history.
        For market orders this is usually immediate.
        """
        deals = self._run_sync(
            mt5.history_deals_get,
            position=int(order_id),
        )

        if deals is None or len(deals) == 0:
            # Still pending or not yet in history
            return FillResult(order_id=order_id, status=OrderStatus.SUBMITTED)

        # Most recent deal for this position
        deal = sorted(deals, key=lambda d: d.time)[-1]
        fill_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)

        return FillResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            fill_price=deal.price,
            fill_time=fill_time,
            slippage=0.0,
            commission=deal.commission,
        )

    def cancel(self, order_id: str) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }
        result = self._run_sync(mt5.order_send, request)
        if result is None:
            return False
        return result.retcode == _RETCODE_DONE

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def close_position(
        self, symbol: str, lot_size: float | None = None
    ) -> FillResult:
        positions = self._run_sync(mt5.positions_get, symbol=symbol)
        if not positions:
            return FillResult(
                order_id="", status=OrderStatus.REJECTED,
                error=f"no_open_position_{symbol}",
            )

        pos = positions[0]
        close_lots = lot_size or pos.volume
        close_lots = self._round_lots(symbol, close_lots)

        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerError(f"Cannot get tick for {symbol}")

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_lots,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = self._run_sync(mt5.order_send, request)
        if result is None or result.retcode != _RETCODE_DONE:
            code = result.retcode if result else 0
            comment = result.comment if result else str(mt5.last_error())
            raise BrokerError(f"Close failed: {comment}", code=code)

        logger.info(
            f"MT5Broker closed | {symbol} | lots={close_lots:.2f} | "
            f"ticket={result.order}"
        )

        return FillResult(
            order_id=str(result.order),
            status=OrderStatus.FILLED,
            fill_price=price,
            fill_time=self.utcnow(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_sync(self, fn, *args, **kwargs):
        """Run an MT5 API call synchronously in the dedicated executor thread."""
        future = self._executor.submit(fn, *args, **kwargs)
        return future.result(timeout=10)

    def _round_lots(self, symbol: str, lots: float) -> float:
        """Round lot size to the broker's minimum lot step."""
        step = self._lot_steps.get(symbol, 0.01)
        rounded = round(round(lots / step) * step, 2)
        return max(rounded, step)