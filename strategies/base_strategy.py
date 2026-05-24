"""
strategies/base_strategy.py
----------------------------
Core abstractions for the strategy layer.

Design rules:
  - A strategy receives ONE feature row (the latest completed bar's features).
  - It returns a Signal or None. Nothing else.
  - No broker calls, no risk checks, no state mutation outside the strategy.
  - Strategies are stateful (they can track their own position/context)
    but that state is internal only.
  - Every strategy must implement: on_bar() and reset().
  - Strategies are independently testable without any live connection.

Signal flow:
  Bar → FeaturePipeline → Strategy.on_bar() → Signal → RiskEngine → Execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Signal direction
# ---------------------------------------------------------------------------

class Direction(Enum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"   # close / exit signal


# ---------------------------------------------------------------------------
# Signal — the atomic output of every strategy
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """
    A trading signal emitted by a strategy.

    Fields:
        symbol      : instrument (e.g. "XAUUSD")
        timeframe   : bar timeframe that generated the signal (e.g. "H1")
        direction   : LONG, SHORT, or FLAT (exit)
        strength    : confidence in [0.0, 1.0].
                      Rule-based strategies use 1.0 (binary).
                      ML strategies use the model's class probability.
        strategy_id : unique name of the originating strategy
        bar_time    : event_time of the bar that triggered the signal
        signal_time : wall-clock time when the signal was generated (UTC)
        stop_loss   : absolute price for stop loss (None = risk engine sets it)
        take_profit : absolute price for take profit (None = risk engine sets it)
        atr_at_signal : ATR value at signal time (used by risk engine for sizing)
        meta        : arbitrary dict for strategy-specific metadata (debug only)
    """
    symbol: str
    timeframe: str
    direction: Direction
    strength: float                     # [0.0, 1.0]
    strategy_id: str
    bar_time: datetime                  # event_time of triggering bar
    signal_time: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    stop_loss: float | None = None
    take_profit: float | None = None
    atr_at_signal: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert 0.0 <= self.strength <= 1.0, \
            f"Signal strength must be in [0, 1], got {self.strength}"
        assert self.bar_time.tzinfo is not None, \
            "bar_time must be UTC-aware"
        assert self.signal_time.tzinfo is not None, \
            "signal_time must be UTC-aware"

    @property
    def is_entry(self) -> bool:
        return self.direction in (Direction.LONG, Direction.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.direction == Direction.FLAT

    def __repr__(self) -> str:
        return (
            f"Signal({self.strategy_id} | {self.symbol}/{self.timeframe} | "
            f"{self.direction.value.upper()} | strength={self.strength:.2f} | "
            f"bar={self.bar_time.strftime('%Y-%m-%d %H:%M')})"
        )


# ---------------------------------------------------------------------------
# Strategy state
# ---------------------------------------------------------------------------

@dataclass
class StrategyState:
    """
    Lightweight internal state a strategy can track between bars.
    Kept separate so it can be serialised for live/restart continuity.
    """
    strategy_id: str
    current_position: Direction = Direction.FLAT
    bars_in_trade: int = 0
    last_signal_time: datetime | None = None
    total_signals: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base for all strategies.

    Subclasses must implement:
      - on_bar(row)  : core signal logic
      - reset()      : clear internal state

    Subclasses may override:
      - warmup_bars  : minimum bars required before emitting signals
      - on_fill()    : called when a previously emitted signal was filled
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        warmup_bars: int = 200,
    ) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.warmup_bars = warmup_bars
        self._state = StrategyState(strategy_id=strategy_id)
        self._bar_count = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_bar(self, row: dict) -> Signal | None:
        """
        Process one feature row (dict of column → value).
        Called on every new completed bar.

        Args:
            row: dict from df.tail(1).to_dicts()[0]
                 Contains all OHLCV fields + all computed features.
                 Values from THIS bar only — no future data.

        Returns:
            Signal if a trade trigger is detected, else None.

        Rules:
          - Never call broker APIs here.
          - Never read external state here.
          - Access only `row` and `self._state`.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state. Called before each backtest run."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional override)
    # ------------------------------------------------------------------

    def on_fill(self, signal: Signal, fill_price: float) -> None:
        """
        Called by the execution layer when a signal was filled.
        Update internal position tracking here.
        """
        self._state.current_position = signal.direction
        self._state.last_signal_time = signal.signal_time

    def on_bar_update(self, row: dict) -> Signal | None:
        """
        Public entry point. Wraps on_bar() with:
          - bar count tracking
          - warmup guard (no signals until warmup_bars seen)
          - None-safety
        """
        self._bar_count += 1
        if self._bar_count < self.warmup_bars:
            return None
        signal = self.on_bar(row)
        if signal is not None:
            self._state.total_signals += 1
        return signal

    # ------------------------------------------------------------------
    # Helpers available to all strategies
    # ------------------------------------------------------------------

    @property
    def state(self) -> StrategyState:
        return self._state

    @property
    def is_flat(self) -> bool:
        return self._state.current_position == Direction.FLAT

    @property
    def is_long(self) -> bool:
        return self._state.current_position == Direction.LONG

    @property
    def is_short(self) -> bool:
        return self._state.current_position == Direction.SHORT

    def _make_signal(
        self,
        row: dict,
        direction: Direction,
        strength: float = 1.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        meta: dict | None = None,
    ) -> Signal:
        """Convenience factory — fills in boilerplate fields."""
        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=direction,
            strength=strength,
            strategy_id=self.strategy_id,
            bar_time=row["event_time"],
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_at_signal=row.get(f"atr_14"),
            meta=meta or {},
        )

    def _get(self, row: dict, key: str, default=None):
        """Safe feature accessor — returns default if key missing or None."""
        v = row.get(key, default)
        return default if v is None else v