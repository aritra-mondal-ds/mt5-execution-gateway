"""
risk/drawdown.py
----------------
Drawdown circuit breaker.

Tracks two independent drawdown measures:
  1. Daily drawdown   : (daily_start_equity - current_equity) / daily_start_equity
                        Resets at the start of each trading day.
  2. Peak drawdown    : (peak_equity - current_equity) / peak_equity
                        Never resets — tracks max drawdown from all-time high.

Trading is halted (is_halted = True) when either limit is breached.
The halt is sticky — once halted, resume() must be called explicitly
(or daily reset occurs for the daily breaker).

Design:
  - Separation of concerns: drawdown tracking has NO knowledge of orders
    or strategies. It simply answers "is_halted?" given current equity.
  - The RiskEngine calls update(equity) on every bar and checks is_halted
    before processing any signal.
  - Daily reset happens automatically when update() is called on a new
    calendar day (UTC).
"""

from __future__ import annotations

from datetime import datetime, timezone, date
from dataclasses import dataclass

from loguru import logger


@dataclass
class DrawdownState:
    peak_equity: float
    daily_start_equity: float
    daily_start_date: date
    current_equity: float
    daily_drawdown: float = 0.0
    peak_drawdown: float = 0.0
    halt_reason: str = ""
    is_halted: bool = False
    total_halts: int = 0


class DrawdownBreaker:
    """
    Monitors equity and halts trading when drawdown limits are breached.

    Usage:
        breaker = DrawdownBreaker(max_daily=0.03, max_peak=0.08)
        breaker.initialize(equity=10_000)

        # On every bar / equity update:
        breaker.update(current_equity=9_800)
        if breaker.is_halted:
            # skip all signals

        # Manual resume (e.g. after human review):
        breaker.resume()
    """

    def __init__(
        self,
        max_daily_drawdown: float = 0.03,    # 3% daily loss limit
        max_peak_drawdown: float = 0.08,     # 8% from peak
        warn_daily_threshold: float = 0.02,  # warn at 2% daily loss
        warn_peak_threshold: float = 0.05,   # warn at 5% from peak
    ) -> None:
        self.max_daily_drawdown = max_daily_drawdown
        self.max_peak_drawdown = max_peak_drawdown
        self.warn_daily_threshold = warn_daily_threshold
        self.warn_peak_threshold = warn_peak_threshold
        self._state: DrawdownState | None = None

    def initialize(self, equity: float, as_of: datetime | None = None) -> None:
        """Call once at system startup with the current account equity."""
        today = (as_of or datetime.now(tz=timezone.utc)).date()
        self._state = DrawdownState(
            peak_equity=equity,
            daily_start_equity=equity,
            daily_start_date=today,
            current_equity=equity,
        )
        logger.info(
            f"DrawdownBreaker initialized | equity={equity:.2f} | "
            f"max_daily={self.max_daily_drawdown:.1%} | "
            f"max_peak={self.max_peak_drawdown:.1%}"
        )

    def update(self, equity: float, as_of: datetime | None = None) -> bool:
        """
        Update equity and check drawdown limits.

        Returns True if trading is now halted (new halt triggered).
        Returns False if status unchanged.

        Always call this before evaluating any signal.
        """
        assert self._state is not None, "Call initialize() first"
        today = (as_of or datetime.now(tz=timezone.utc)).date()

        # Daily reset: new day → reset daily baseline (but keep peak)
        if today > self._state.daily_start_date:
            self._daily_reset(equity, today)

        self._state.current_equity = equity

        # Update peak
        if equity > self._state.peak_equity:
            self._state.peak_equity = equity

        # Calculate drawdowns
        daily_dd = (
            (self._state.daily_start_equity - equity) /
            self._state.daily_start_equity
        ) if self._state.daily_start_equity > 0 else 0.0

        peak_dd = (
            (self._state.peak_equity - equity) /
            self._state.peak_equity
        ) if self._state.peak_equity > 0 else 0.0

        self._state.daily_drawdown = max(0.0, daily_dd)
        self._state.peak_drawdown = max(0.0, peak_dd)

        # Warnings
        if self._state.daily_drawdown >= self.warn_daily_threshold and \
                not self._state.is_halted:
            logger.warning(
                f"Daily drawdown warning: {self._state.daily_drawdown:.2%} "
                f"(limit: {self.max_daily_drawdown:.2%})"
            )
        if self._state.peak_drawdown >= self.warn_peak_threshold and \
                not self._state.is_halted:
            logger.warning(
                f"Peak drawdown warning: {self._state.peak_drawdown:.2%} "
                f"(limit: {self.max_peak_drawdown:.2%})"
            )

        # Circuit breaker checks
        if not self._state.is_halted:
            if self._state.daily_drawdown >= self.max_daily_drawdown:
                self._halt(
                    f"daily_drawdown_limit | "
                    f"{self._state.daily_drawdown:.2%} >= {self.max_daily_drawdown:.2%}"
                )
                return True

            if self._state.peak_drawdown >= self.max_peak_drawdown:
                self._halt(
                    f"peak_drawdown_limit | "
                    f"{self._state.peak_drawdown:.2%} >= {self.max_peak_drawdown:.2%}"
                )
                return True

        return False

    def resume(self, reason: str = "manual") -> None:
        """Manually lift a halt (e.g. after human review)."""
        assert self._state is not None
        if self._state.is_halted:
            logger.info(
                f"DrawdownBreaker: halt lifted | reason={reason} | "
                f"equity={self._state.current_equity:.2f}"
            )
            self._state.is_halted = False
            self._state.halt_reason = ""

    @property
    def is_halted(self) -> bool:
        if self._state is None:
            return True  # not initialized → safe default
        return self._state.is_halted

    @property
    def state(self) -> DrawdownState | None:
        return self._state

    @property
    def daily_drawdown(self) -> float:
        return self._state.daily_drawdown if self._state else 0.0

    @property
    def peak_drawdown(self) -> float:
        return self._state.peak_drawdown if self._state else 0.0

    def _halt(self, reason: str) -> None:
        assert self._state is not None
        self._state.is_halted = True
        self._state.halt_reason = reason
        self._state.total_halts += 1
        logger.critical(
            f"TRADING HALTED | {reason} | "
            f"equity={self._state.current_equity:.2f} | "
            f"peak={self._state.peak_equity:.2f}"
        )

    def _daily_reset(self, equity: float, today: date) -> None:
        assert self._state is not None
        logger.info(
            f"DrawdownBreaker: daily reset | "
            f"new_day={today} | start_equity={equity:.2f}"
        )
        self._state.daily_start_equity = equity
        self._state.daily_start_date = today
        self._state.daily_drawdown = 0.0
        # Daily halt auto-lifts on new day; peak halt stays
        if self._state.is_halted and "daily" in self._state.halt_reason:
            self._state.is_halted = False
            self._state.halt_reason = ""