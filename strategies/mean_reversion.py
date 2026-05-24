"""
strategies/mean_reversion.py
-----------------------------
Bollinger Band mean reversion with RSI confirmation.

Logic:
  LONG  entry : close < bb_lower  AND  rsi < oversold_threshold
                (price stretched below band + momentum oversold)
  SHORT entry : close > bb_upper  AND  rsi > overbought_threshold
                (price stretched above band + momentum overbought)
  EXIT        : close crosses back through bb_mid (mean reversion complete)
                OR RSI returns to neutral zone
                OR ATR stop hit

This strategy is counter-trend — it bets on reversion, not continuation.
It works best in ranging/low-ADX markets (opposite of TrendStrategy).
Combining the two creates a regime-adaptive portfolio.

Key safeguards:
  - Requires close to be OUTSIDE the band (not just touching it)
  - RSI threshold confirmation prevents catching falling knives in trends
  - ADX ceiling filter: skip entry if ADX > adx_ceiling (trending = dangerous)
  - Max holding period: force exit after max_bars_in_trade bars
"""

from __future__ import annotations

from loguru import logger

from strategies.base_strategy import BaseStrategy, Direction, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band + RSI mean reversion.

    Required feature columns:
      bb_lower_{bb_period}   e.g. bb_lower_20
      bb_upper_{bb_period}   e.g. bb_upper_20
      bb_mid_{bb_period}     e.g. bb_mid_20
      rsi_{rsi_period}       e.g. rsi_14
      adx_{adx_period}       e.g. adx_14
      atr_{atr_period}       e.g. atr_14
      close, event_time
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        bb_period: int = 20,
        rsi_period: int = 14,
        adx_period: int = 14,
        atr_period: int = 14,
        oversold_rsi: float = 35.0,
        overbought_rsi: float = 65.0,
        adx_ceiling: float = 30.0,       # don't mean-revert into a strong trend
        atr_multiplier: float = 2.0,     # stop loss distance
        tp_multiplier: float = 1.0,      # TP at band midline by default (set None)
        use_mid_as_tp: bool = True,      # TP = BB midline (the mean)
        max_bars_in_trade: int = 20,     # force exit after N bars
        min_band_width: float = 0.001,   # skip if bands too tight (low vol)
    ) -> None:
        super().__init__(
            strategy_id=f"mean_rev_bb{bb_period}_rsi{rsi_period}",
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=max(bb_period + 10, rsi_period + 10, 50),
        )
        self._bb_lower = f"bb_lower_{bb_period}"
        self._bb_upper = f"bb_upper_{bb_period}"
        self._bb_mid   = f"bb_mid_{bb_period}"
        self._bb_width = f"bb_width_{bb_period}"
        self._rsi = f"rsi_{rsi_period}"
        self._adx = f"adx_{adx_period}"
        self._atr = f"atr_{atr_period}"

        self.oversold_rsi = oversold_rsi
        self.overbought_rsi = overbought_rsi
        self.adx_ceiling = adx_ceiling
        self.atr_multiplier = atr_multiplier
        self.tp_multiplier = tp_multiplier
        self.use_mid_as_tp = use_mid_as_tp
        self.max_bars_in_trade = max_bars_in_trade
        self.min_band_width = min_band_width

        self._entry_price: float | None = None
        self._stop_loss: float | None = None
        self._take_profit: float | None = None

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def on_bar(self, row: dict) -> Signal | None:
        close     = self._get(row, "close")
        bb_lower  = self._get(row, self._bb_lower)
        bb_upper  = self._get(row, self._bb_upper)
        bb_mid    = self._get(row, self._bb_mid)
        bb_width  = self._get(row, self._bb_width)
        rsi       = self._get(row, self._rsi)
        adx       = self._get(row, self._adx)
        atr       = self._get(row, self._atr)

        if any(v is None for v in [close, bb_lower, bb_upper, bb_mid, rsi, atr]):
            return None

        # --- Exit logic first ---
        if not self.is_flat:
            signal = self._check_exit(row, close, bb_mid, rsi, atr)
            if signal is not None:
                self._reset_trade_state()
                self._state.current_position = Direction.FLAT
                return signal
            self._state.bars_in_trade += 1

        # --- Entry logic ---
        if self.is_flat:
            # Skip if bands are too tight (low volatility — little edge)
            if bb_width is not None and bb_width < self.min_band_width:
                return None
            # Skip if in a strong trend (dangerous to counter-trade)
            if adx is not None and adx > self.adx_ceiling:
                return None

            signal = self._check_entry(row, close, bb_lower, bb_upper, bb_mid, rsi, atr)
            if signal is not None:
                self._entry_price = close
                self._stop_loss = signal.stop_loss
                self._take_profit = signal.take_profit
                self._state.bars_in_trade = 0
                self._state.current_position = signal.direction
                return signal   # ← return before is_flat becomes False

        return None

    def _check_entry(
        self,
        row: dict,
        close: float,
        bb_lower: float,
        bb_upper: float,
        bb_mid: float,
        rsi: float,
        atr: float,
    ) -> Signal | None:
        stop = atr * self.atr_multiplier

        # Long: price below lower band + RSI oversold
        if close < bb_lower and rsi < self.oversold_rsi:
            sl = close - stop
            tp = bb_mid if self.use_mid_as_tp else close + (atr * self.tp_multiplier)
            logger.debug(
                f"{self.strategy_id} LONG | {row['event_time']} | "
                f"close={close:.4f} BB_lower={bb_lower:.4f} RSI={rsi:.1f}"
            )
            return self._make_signal(
                row, Direction.LONG,
                stop_loss=sl, take_profit=tp,
                meta={"rsi": rsi, "bb_lower": bb_lower, "setup": "oversold"},
            )

        # Short: price above upper band + RSI overbought
        if close > bb_upper and rsi > self.overbought_rsi:
            sl = close + stop
            tp = bb_mid if self.use_mid_as_tp else close - (atr * self.tp_multiplier)
            logger.debug(
                f"{self.strategy_id} SHORT | {row['event_time']} | "
                f"close={close:.4f} BB_upper={bb_upper:.4f} RSI={rsi:.1f}"
            )
            return self._make_signal(
                row, Direction.SHORT,
                stop_loss=sl, take_profit=tp,
                meta={"rsi": rsi, "bb_upper": bb_upper, "setup": "overbought"},
            )

        return None

    def _check_exit(
        self,
        row: dict,
        close: float,
        bb_mid: float,
        rsi: float,
        atr: float,
    ) -> Signal | None:
        exit_reason = None

        # Max holding period
        if self._state.bars_in_trade >= self.max_bars_in_trade:
            exit_reason = "max_holding_period"

        # Long exit: price crossed back to midline OR RSI recovered
        elif self.is_long:
            if close >= bb_mid:
                exit_reason = "reverted_to_mean"
            elif rsi > 50:
                exit_reason = "rsi_recovered"

        # Short exit: price crossed back to midline OR RSI recovered
        elif self.is_short:
            if close <= bb_mid:
                exit_reason = "reverted_to_mean"
            elif rsi < 50:
                exit_reason = "rsi_recovered"

        if exit_reason:
            logger.debug(
                f"{self.strategy_id} EXIT | {row['event_time']} | "
                f"reason={exit_reason} close={close:.4f}"
            )
            return self._make_signal(
                row, Direction.FLAT,
                meta={"exit_reason": exit_reason, "bars_held": self._state.bars_in_trade},
            )

        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_trade_state(self) -> None:
        self._entry_price = None
        self._stop_loss = None
        self._take_profit = None
        self._state.bars_in_trade = 0

    def reset(self) -> None:
        from strategies.base_strategy import StrategyState
        self._state = StrategyState(strategy_id=self.strategy_id)
        self._bar_count = 0
        self._reset_trade_state()