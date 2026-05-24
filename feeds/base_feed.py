"""
feeds/base_feed.py
------------------
Abstract interface that every data feed must implement.

Design rules:
  - Feeds emit canonical Bar objects only. No raw broker structs leak out.
  - Every Bar carries both event_time (when the bar closed in the market)
    and publish_time (when our system received/processed it).
    These two timestamps are the foundation of point-in-time correctness.
  - Feeds are async-first. Synchronous adapters wrap async internally.
  - A feed does ONE thing: deliver bars. No feature computation here.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Awaitable


# ---------------------------------------------------------------------------
# Canonical data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """
    A single OHLCV bar — the atomic unit of market data in this system.

    Frozen + slots: immutable after creation, minimal memory footprint.
    Timestamps are always UTC-aware datetime objects.

    event_time   : when this bar *closed* in the market (the candle's timestamp)
    publish_time : when *our system* saw/processed this bar
                   In live trading these differ by network latency.
                   In backtesting, publish_time = event_time (no lookahead).
    """
    symbol: str
    timeframe: str                   # e.g. "M1", "H1", "1m", "1h"
    event_time: datetime             # bar close time, UTC
    publish_time: datetime           # system receipt time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0              # in points/pips, 0 if unknown
    source: str = ""                 # feed identifier, e.g. "mt5", "binance"

    def __post_init__(self) -> None:
        # Enforce UTC timezone on both timestamps
        assert self.event_time.tzinfo is not None, \
            "event_time must be timezone-aware (UTC)"
        assert self.publish_time.tzinfo is not None, \
            "publish_time must be timezone-aware (UTC)"
        assert self.high >= self.low, \
            f"high ({self.high}) must be >= low ({self.low})"
        assert self.high >= self.open >= 0, \
            "open must be positive and <= high"
        assert self.high >= self.close >= 0, \
            "close must be positive and <= high"
        assert self.volume >= 0, "volume must be non-negative"

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        """Serialise to a flat dict (for Parquet writes)."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "event_time": self.event_time,
            "publish_time": self.publish_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "spread": self.spread,
            "source": self.source,
        }


@dataclass
class FeedStatus:
    """Health / diagnostic snapshot of a running feed."""
    feed_id: str
    is_connected: bool = False
    last_bar_time: datetime | None = None
    bars_received: int = 0
    errors: int = 0
    last_error: str = ""


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------
BarCallback = Callable[[Bar], Awaitable[None]]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseFeed(ABC):
    """
    Every feed (MT5, crypto, forex REST) inherits from this.

    Lifecycle:
        await feed.connect()
        async for bar in feed.stream(symbols, timeframes):
            process(bar)
        await feed.disconnect()

    Alternatively, register callbacks:
        feed.on_bar(my_async_handler)
        await feed.start()   # runs until cancelled
    """

    def __init__(self, feed_id: str) -> None:
        self.feed_id = feed_id
        self._status = FeedStatus(feed_id=feed_id)
        self._bar_callbacks: list[BarCallback] = []
        self._running = False

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every concrete feed
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source. Raise on failure."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
    ) -> list[Bar]:
        """
        Fetch the last `n_bars` closed bars for a symbol/timeframe.
        Used for warmup on startup and backtest data loading.
        Returns bars in ascending time order (oldest first).
        """
        ...

    @abstractmethod
    async def stream(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> AsyncIterator[Bar]:
        """
        Yield bars as they arrive (live) or from a replay (backtest).
        Each yielded Bar has a publish_time = utcnow() in live mode.
        """
        ...

    # ------------------------------------------------------------------
    # Callback registration (alternative to async-for usage)
    # ------------------------------------------------------------------

    def on_bar(self, callback: BarCallback) -> None:
        """Register an async callback invoked on each new bar."""
        self._bar_callbacks.append(callback)

    async def _emit(self, bar: Bar) -> None:
        """Internal: update status and fire all registered callbacks."""
        self._status.bars_received += 1
        self._status.last_bar_time = bar.publish_time
        for cb in self._bar_callbacks:
            try:
                await cb(bar)
            except Exception as exc:        # noqa: BLE001
                self._status.errors += 1
                self._status.last_error = str(exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> FeedStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status.is_connected

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BaseFeed":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def utcnow() -> datetime:
        """Always use this for publish_time — enforces UTC."""
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert a naive or offset-aware datetime to UTC-aware."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)