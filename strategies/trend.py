"""
strategies/trend.py
--------------------
EMA crossover trend-following strategy with ADX trend-strength filter.

Logic:
  LONG  entry : fast EMA crosses above slow EMA  AND  ADX > adx_threshold
  SHORT entry : fast EMA crosses below slow EMA  AND  ADX > adx_threshold
  EXIT        : opposite crossover OR ADX drops below exit_adx_threshold

Stop loss  : entry_price ± atr_multiplier * ATR(14)
Take profit: entry_price ± tp_multiplier * ATR(14)   (risk:reward ratio)

Design:
  - Uses PREVIOUS bar's crossover signal to avoid same-bar lookahead.
    i.e. we act on the bar AFTER the crossover is confirmed.
  - ATR-based stops normalise risk across different volatility regimes.
  - ADX filter prevents trading in choppy/ranging markets.
  - One position at a time per instance. Multiple instances can run
    different parameter sets in parallel (via the portfolio allocator).
"""

from __future__ import annotations

from loguru import logger

from strategies.base_strategy import BaseStrategy, Direction, Signal


class TrendStrategy(BaseStrategy):
    """
    EMA crossover with ADX trend filter.

    Required feature columns (from pipeline):
      ema_{fast_period}     e.g. ema_9
      ema_{slow_period}     e.g. ema_21
      adx_{adx_period}      e.g. adx_14
      atr_{atr_period}      e.g. atr_14
      close
      event_time
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        fast_period: int = 9,
        slow_period: int = 21,
        adx_period: int = 14,
        atr_period: int = 14,
        adx_threshold: float = 25.0,       # min ADX to enter
        exit_adx_threshold: float = 20.0,  # ADX below this → forced exit
        atr_multiplier: float = 1.5,       # stop loss distance in ATR
        tp_multiplier: float = 3.0,        # take profit distance in ATR
        cooldown_bars: int = 3,            # bars to wait after an exit
    ) -> None:
        super().__init__(
            strategy_id=f"trend_ema{fast_period}_{slow_period}",
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=max(slow_period + 10, 50),
        )
        self._fast = f"ema_{fast_period}"
        self._slow = f"ema_{slow_period}"
        self._adx = f"adx_{adx_period}"
        self._atr = f"atr_{atr_period}"
        self.adx_threshold = adx_threshold
        self.exit_adx_threshold = exit_adx_threshold
        self.atr_multiplier = atr_multiplier
        self.tp_multiplier = tp_multiplier
        self.cooldown_bars = cooldown_bars

        # Internal state beyond StrategyState
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._cooldown_counter: int = 0
        self._entry_price: float | None = None

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def on_bar(self, row: dict) -> Signal | None:
        fast = self._get(row, self._fast)
        slow = self._get(row, self._slow)
        adx  = self._get(row, self._adx)
        atr  = self._get(row, self._atr)
        close = self._get(row, "close")

        # Can't act without all required features
        if any(v is None for v in [fast, slow, adx, atr, close]):
            return None

        signal = None

        # --- Exit logic (checked before entry) ---
        if not self.is_flat:
            signal = self._check_exit(row, fast, slow, adx, close, atr)
            if signal is not None:
                self._cooldown_counter = self.cooldown_bars
                self._entry_price = None
                self._state.current_position = Direction.FLAT
                self._prev_fast = fast
                self._prev_slow = slow
                return signal

        # --- Cooldown ---
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._prev_fast = fast
            self._prev_slow = slow
            return None

        # --- Entry logic ---
        if self.is_flat and self._prev_fast is not None:
            signal = self._check_entry(row, fast, slow, adx, close, atr)
            if signal is not None:
                self._entry_price = close
                self._state.current_position = signal.direction

        self._prev_fast = fast
        self._prev_slow = slow
        return signal

    def _check_entry(
        self,
        row: dict,
        fast: float,
        slow: float,
        adx: float,
        close: float,
        atr: float,
    ) -> Signal | None:
        """
        Entry: crossover on PREVIOUS bar confirmed by ADX filter.

        Crossover is detected by comparing current and previous EMA spread.
        prev_fast < prev_slow and now fast > slow → bullish crossover.
        """
        assert self._prev_fast is not None
        assert self._prev_slow is not None

        prev_spread = self._prev_fast - self._prev_slow
        curr_spread = fast - slow

        if adx < self.adx_threshold:
            return None  # choppy market — stay out

        stop = atr * self.atr_multiplier
        tp = atr * self.tp_multiplier

        # Bullish crossover
        if prev_spread <= 0 and curr_spread > 0:
            sl_price = close - stop
            tp_price = close + tp
            logger.debug(
                f"{self.strategy_id} LONG | {row['event_time']} | "
                f"close={close:.4f} SL={sl_price:.4f} TP={tp_price:.4f} ADX={adx:.1f}"
            )
            return self._make_signal(
                row, Direction.LONG,
                stop_loss=sl_price,
                take_profit=tp_price,
                meta={"adx": adx, "fast": fast, "slow": slow},
            )

        # Bearish crossover
        if prev_spread >= 0 and curr_spread < 0:
            sl_price = close + stop
            tp_price = close - tp
            logger.debug(
                f"{self.strategy_id} SHORT | {row['event_time']} | "
                f"close={close:.4f} SL={sl_price:.4f} TP={tp_price:.4f} ADX={adx:.1f}"
            )
            return self._make_signal(
                row, Direction.SHORT,
                stop_loss=sl_price,
                take_profit=tp_price,
                meta={"adx": adx, "fast": fast, "slow": slow},
            )

        return None

    def _check_exit(
        self,
        row: dict,
        fast: float,
        slow: float,
        adx: float,
        close: float,
        atr: float,
    ) -> Signal | None:
        """
        Exit conditions:
          1. Opposite crossover (trend reversal)
          2. ADX drops below exit_adx_threshold (trend exhaustion)
          3. Stop loss / take profit hit (handled by execution layer,
             but we track for state consistency)
        """
        prev_spread = (self._prev_fast or fast) - (self._prev_slow or slow)
        curr_spread = fast - slow

        exit_reason = None

        if self.is_long:
            if curr_spread < 0 and prev_spread >= 0:
                exit_reason = "crossover_reversal"
            elif adx < self.exit_adx_threshold:
                exit_reason = "adx_exhaustion"

        elif self.is_short:
            if curr_spread > 0 and prev_spread <= 0:
                exit_reason = "crossover_reversal"
            elif adx < self.exit_adx_threshold:
                exit_reason = "adx_exhaustion"

        if exit_reason:
            logger.debug(
                f"{self.strategy_id} EXIT | {row['event_time']} | "
                f"reason={exit_reason} close={close:.4f}"
            )
            return self._make_signal(
                row, Direction.FLAT,
                meta={"exit_reason": exit_reason, "adx": adx},
            )

        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        from strategies.base_strategy import StrategyState
        self._state = StrategyState(strategy_id=self.strategy_id)
        self._bar_count = 0
        self._prev_fast = None
        self._prev_slow = None
        self._cooldown_counter = 0
        self._entry_price = None