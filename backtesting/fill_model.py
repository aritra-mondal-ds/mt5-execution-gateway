"""
backtesting/fill_model.py
--------------------------
Realistic bar-level fill model for the event-driven backtester.

This is the primary fix for backtest realism. Most vectorised backtests
assume fills at the close of the signal bar — which is impossible in
reality. This model simulates what actually happens inside each bar.

Bar processing order (strict):
  1. Check if any open position's SL was hit  (high/low of bar)
  2. Check if any open position's TP was hit  (high/low of bar)
  3. Process new signals → open new positions at next bar's open
     (we cannot act on a signal from bar T until bar T+1)

SL/TP hit detection:
  For a LONG position:
    SL hit if bar.low  <= stop_loss   → fill at stop_loss (with gap penalty)
    TP hit if bar.high >= take_profit → fill at take_profit

  For a SHORT position:
    SL hit if bar.high >= stop_loss   → fill at stop_loss (with gap penalty)
    TP hit if bar.low  <= take_profit → fill at take_profit

  If BOTH SL and TP are hit in the same bar (volatile bar):
    We conservatively assume SL was hit first (worst case).
    This is realistic: a wide-ranging bar likely swept the stop first.

Gap fill model:
  In real markets, gaps occur overnight or on news.
  If a bar opens beyond the SL price:
    fill_price = bar.open (the gap price, not the SL price)
  This means stop losses can be filled significantly worse than the
  requested SL — which is a critical realism feature.

Limit orders:
  Require price to trade THROUGH the limit (not just touch).
  This prevents the optimistic assumption that every limit order
  fills at its exact price on the first bar it's reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from feeds.base_feed import Bar
from strategies.base_strategy import Direction


class FillType(Enum):
    STOP_LOSS    = "stop_loss"
    TAKE_PROFIT  = "take_profit"
    SIGNAL       = "signal"      # new entry fill
    GAP          = "gap"         # gapped through SL


@dataclass
class BarFill:
    """Result of checking one open position against one bar."""
    triggered: bool
    fill_type: FillType | None = None
    fill_price: float | None = None
    slippage: float = 0.0


@dataclass
class OpenPosition:
    """
    Lightweight position representation used only by the fill model.
    The backtesting engine maintains these in its internal ledger.
    """
    symbol: str
    direction: Direction
    entry_price: float
    lot_size: float
    stop_loss: float | None
    take_profit: float | None
    entry_bar_index: int       # which bar index this was opened on
    strategy_id: str


class FillModel:
    """
    Determines how bars interact with open positions and pending orders.

    Usage (called by the engine on every bar):
        fill_model = FillModel(slippage_factor=0.1, gap_fill=True)

        for position in open_positions:
            result = fill_model.check_position(bar, position)
            if result.triggered:
                engine.close_position(position, result.fill_price)
    """

    def __init__(
        self,
        slippage_factor: float = 0.1,    # fraction of spread added to SL fill
        gap_fill: bool = True,            # fill at open if bar gaps through SL
        require_price_through: bool = True, # limit orders need full cross
        spread_points: dict[str, float] | None = None,
    ) -> None:
        self.slippage_factor = slippage_factor
        self.gap_fill = gap_fill
        self.require_price_through = require_price_through
        self._spreads = spread_points or {
            "XAUUSD": 0.30, "EURUSD": 0.00010,
            "GBPUSD": 0.00013, "BTCUSDT": 10.0,
        }

    def check_position(self, bar: Bar, pos: OpenPosition) -> BarFill:
        """
        Check whether a bar triggers an SL or TP for an open position.

        Conservative rule: if both SL and TP are hit in the same bar,
        SL is assumed to have triggered first.

        Returns BarFill(triggered=False) if neither is hit.
        """
        if pos.direction == Direction.LONG:
            return self._check_long(bar, pos)
        elif pos.direction == Direction.SHORT:
            return self._check_short(bar, pos)
        return BarFill(triggered=False)

    def _check_long(self, bar: Bar, pos: OpenPosition) -> BarFill:
        sl = pos.stop_loss
        tp = pos.take_profit

        # Gap open: bar opens below SL → fill at open (worst case)
        if self.gap_fill and sl is not None and bar.open < sl:
            slippage = self._slippage(bar.symbol)
            return BarFill(
                triggered=True,
                fill_type=FillType.GAP,
                fill_price=bar.open - slippage,
                slippage=slippage,
            )

        # SL hit: bar trades down to or through stop_loss
        if sl is not None and bar.low <= sl:
            slippage = self._slippage(bar.symbol)
            return BarFill(
                triggered=True,
                fill_type=FillType.STOP_LOSS,
                fill_price=sl - slippage,   # fill slightly below SL
                slippage=slippage,
            )

        # TP hit: bar trades up to take_profit
        if tp is not None:
            cross = bar.high > tp if self.require_price_through else bar.high >= tp
            if cross:
                return BarFill(
                    triggered=True,
                    fill_type=FillType.TAKE_PROFIT,
                    fill_price=tp,
                    slippage=0.0,
                )

        return BarFill(triggered=False)

    def _check_short(self, bar: Bar, pos: OpenPosition) -> BarFill:
        sl = pos.stop_loss
        tp = pos.take_profit

        # Gap open: bar opens above SL → fill at open (worst case)
        if self.gap_fill and sl is not None and bar.open > sl:
            slippage = self._slippage(bar.symbol)
            return BarFill(
                triggered=True,
                fill_type=FillType.GAP,
                fill_price=bar.open + slippage,
                slippage=slippage,
            )

        # SL hit: bar trades up to or through stop_loss
        if sl is not None and bar.high >= sl:
            slippage = self._slippage(bar.symbol)
            return BarFill(
                triggered=True,
                fill_type=FillType.STOP_LOSS,
                fill_price=sl + slippage,
                slippage=slippage,
            )

        # TP hit: bar trades down to take_profit
        if tp is not None:
            cross = bar.low < tp if self.require_price_through else bar.low <= tp
            if cross:
                return BarFill(
                    triggered=True,
                    fill_type=FillType.TAKE_PROFIT,
                    fill_price=tp,
                    slippage=0.0,
                )

        return BarFill(triggered=False)

    def entry_fill_price(self, bar: Bar, direction: Direction) -> float:
        """
        Price at which a new position opens on bar T+1.

        We use the open of the next bar (realistic: signal was on bar T,
        we enter at bar T+1 open). The caller passes the NEXT bar here.
        Spread is added for long entries (we pay the ask).
        """
        spread = self._spreads.get(bar.symbol, 0.0)
        if direction == Direction.LONG:
            return bar.open + spread / 2
        elif direction == Direction.SHORT:
            return bar.open - spread / 2
        return bar.open

    def _slippage(self, symbol: str) -> float:
        spread = self._spreads.get(symbol, 0.0)
        return spread * self.slippage_factor