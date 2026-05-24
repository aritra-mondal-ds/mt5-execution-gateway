"""
feeds/reconnect.py
------------------
Reconnection wrapper for any BaseFeed implementation.

Design:
  ReconnectingFeed wraps any BaseFeed and adds:
    1. Exponential backoff on connect() failures
    2. Stale feed detection (no bars for N seconds → reconnect)
    3. Circuit breaker (stop retrying after max_attempts, fire alert)
    4. Callbacks: on_reconnect, on_circuit_open

  The wrapped feed's interface is unchanged — the rest of the system
  never knows a reconnection happened.

Backoff formula:
  wait = min(base_delay * (2 ** attempt) + jitter, max_delay)
  jitter = random.uniform(0, jitter_factor * base_delay)

  Attempt 0: ~1s, Attempt 1: ~2s, Attempt 2: ~4s … capped at max_delay.
  Jitter prevents thundering-herd if multiple feeds reconnect simultaneously.

Circuit breaker:
  After max_attempts consecutive failures, the breaker OPENS.
  No further connection attempts are made until reset() is called.
  The AlertManager is notified so the operator can intervene.

Stale feed detection:
  A separate asyncio task checks every `health_check_interval` seconds
  whether a new bar has been received. If `max_stale_seconds` passes
  with no bar, the feed is considered dead and reconnection is triggered.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable

from loguru import logger

from feeds.base_feed import BaseFeed, Bar


# ---------------------------------------------------------------------------
# Backoff calculator
# ---------------------------------------------------------------------------

def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.25,
) -> float:
    """
    Calculate wait time for attempt N with full jitter.

    Returns a float seconds value to sleep before the next attempt.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, jitter_factor * delay)
    return delay + jitter


# ---------------------------------------------------------------------------
# Circuit breaker state
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Simple three-state circuit breaker: CLOSED → OPEN → HALF-OPEN.

    CLOSED   : normal operation, connection attempts allowed
    OPEN     : max failures reached, attempts blocked
    HALF-OPEN: one trial attempt allowed after reset_after_seconds
    """

    def __init__(
        self,
        max_failures: int = 10,
        reset_after_seconds: float = 300.0,
    ) -> None:
        self.max_failures = max_failures
        self.reset_after_seconds = reset_after_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._state = "closed"

    @property
    def is_open(self) -> bool:
        if self._state != "open":
            return False
        # Auto-transition to half-open after timeout
        if (self._opened_at is not None and
                time.monotonic() - self._opened_at >= self.reset_after_seconds):
            self._state = "half-open"
            logger.info("CircuitBreaker: HALF-OPEN — trial attempt allowed")
            return False
        return True

    @property
    def state(self) -> str:
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.max_failures and self._state == "closed":
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.critical(
                f"CircuitBreaker OPENED after {self._failures} failures"
            )

    def reset(self) -> None:
        """Manually reset the breaker (called by operator after intervention)."""
        self._failures = 0
        self._state = "closed"
        self._opened_at = None
        logger.info("CircuitBreaker: manually RESET to CLOSED")

    @property
    def failure_count(self) -> int:
        return self._failures


# ---------------------------------------------------------------------------
# Reconnecting feed wrapper
# ---------------------------------------------------------------------------

ReconnectCallback = Callable[[int], Awaitable[None]]  # (attempt_number) -> None


class ReconnectingFeed:
    """
    Wraps any BaseFeed with automatic reconnection.

    Usage:
        feed = MT5Feed(login=..., password=..., server=...)
        reconnecting = ReconnectingFeed(
            feed=feed,
            max_attempts=10,
            base_delay=1.0,
            max_delay=60.0,
            max_stale_seconds=120.0,
        )

        async with reconnecting:
            bars = await reconnecting.fetch_historical("XAUUSD", "H1", 300)
            async for bar in reconnecting.stream(["XAUUSD"], ["H1"]):
                process(bar)

        # Manual circuit reset after operator intervention:
        reconnecting.reset_circuit()
    """

    def __init__(
        self,
        feed: BaseFeed,
        max_attempts: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.25,
        max_stale_seconds: float = 120.0,
        health_check_interval: float = 30.0,
        circuit_reset_after: float = 300.0,
        on_reconnect: ReconnectCallback | None = None,
        on_circuit_open: Callable[[], None] | None = None,
    ) -> None:
        self._feed = feed
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter_factor
        self._max_stale = max_stale_seconds
        self._health_interval = health_check_interval
        self._on_reconnect = on_reconnect
        self._on_circuit_open = on_circuit_open

        self._breaker = CircuitBreaker(
            max_failures=max_attempts,
            reset_after_seconds=circuit_reset_after,
        )
        self._last_bar_time: float = time.monotonic()
        self._reconnect_count: int = 0
        self._health_task: asyncio.Task | None = None
        self._running: bool = False
        self._stream_symbols: list[str] = []
        self._stream_timeframes: list[str] = []

    # ------------------------------------------------------------------
    # Connection with retry
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect with exponential backoff.
        Raises ConnectionError if circuit breaker is open or max attempts exceeded.
        """
        if self._breaker.is_open:
            raise ConnectionError(
                f"Circuit breaker OPEN after {self._breaker.failure_count} failures. "
                "Call reset_circuit() after fixing the underlying issue."
            )

        for attempt in range(self._max_attempts):
            try:
                await self._feed.connect()
                self._breaker.record_success()
                self._last_bar_time = time.monotonic()
                logger.info(
                    f"ReconnectingFeed: connected "
                    f"(attempt {attempt + 1}, reconnects={self._reconnect_count})"
                )
                return

            except Exception as exc:
                self._breaker.record_failure()
                wait = exponential_backoff(
                    attempt, self._base_delay, self._max_delay, self._jitter
                )
                logger.warning(
                    f"ReconnectingFeed: connect failed (attempt {attempt + 1}/"
                    f"{self._max_attempts}): {exc} — retrying in {wait:.1f}s"
                )

                if self._breaker.is_open:
                    if self._on_circuit_open:
                        self._on_circuit_open()
                    raise ConnectionError(
                        f"Circuit breaker OPENED after {attempt + 1} attempts"
                    ) from exc

                await asyncio.sleep(wait)

        raise ConnectionError(
            f"Failed to connect after {self._max_attempts} attempts"
        )

    async def disconnect(self) -> None:
        self._running = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        try:
            await self._feed.disconnect()
        except Exception as e:
            logger.warning(f"ReconnectingFeed: disconnect error: {e}")

    # ------------------------------------------------------------------
    # Fetch historical (with single retry)
    # ------------------------------------------------------------------

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
    ) -> list[Bar]:
        try:
            return await self._feed.fetch_historical(symbol, timeframe, n_bars)
        except Exception as exc:
            logger.warning(
                f"ReconnectingFeed: fetch_historical failed: {exc} — "
                "reconnecting and retrying once"
            )
            await self._reconnect()
            return await self._feed.fetch_historical(symbol, timeframe, n_bars)

    # ------------------------------------------------------------------
    # Stream with auto-reconnection
    # ------------------------------------------------------------------

    async def stream(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> AsyncIterator[Bar]:
        """
        Stream bars. Automatically reconnects on any error.

        On reconnection:
          1. Waits with exponential backoff
          2. Re-calls feed.connect()
          3. Resumes streaming from the point of failure
          4. Fires on_reconnect callback if configured

        Bars are never duplicated across reconnections — the underlying
        MT5Feed's _last_seen tracker handles deduplication.
        """
        self._stream_symbols = symbols
        self._stream_timeframes = timeframes
        self._running = True

        # Start health monitor
        self._health_task = asyncio.create_task(self._health_monitor(symbols, timeframes))

        consecutive_errors = 0

        try:
            while self._running:
                try:
                    async for bar in self._feed.stream(symbols, timeframes):
                        if not self._running:
                            return
                        self._last_bar_time = time.monotonic()
                        consecutive_errors = 0
                        yield bar

                    # Inner async-for exited cleanly (feed exhausted, not an
                    # error). In production MT5Feed streams forever; clean exit
                    # only happens in tests or graceful shutdown — stop looping.
                    return

                except asyncio.CancelledError:
                    return

                except Exception as exc:
                    consecutive_errors += 1
                    logger.warning(
                        f"ReconnectingFeed: stream error "
                        f"(consecutive={consecutive_errors}): {exc}"
                    )

                    if not self._running:
                        return

                    try:
                        await self._reconnect()
                    except ConnectionError as ce:
                        logger.critical(f"ReconnectingFeed: cannot reconnect: {ce}")
                        return

        finally:
            # Always cancel the health-monitor task when stream() exits,
            # regardless of how (clean return, exception, or disconnect).
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> None:
        """
        Async task that detects stale feeds and triggers reconnection.

        A stale feed is one where no bar has been received for
        `max_stale_seconds`. This catches silent failures where the
        TCP connection stays open but no data flows.
        """
        while self._running:
            await asyncio.sleep(self._health_interval)

            stale = time.monotonic() - self._last_bar_time
            if stale > self._max_stale:
                logger.warning(
                    f"ReconnectingFeed: feed stale for {stale:.0f}s "
                    f"(limit={self._max_stale}s) — forcing reconnect"
                )
                try:
                    await self._feed.disconnect()
                except Exception:
                    pass
                try:
                    await self._reconnect()
                    self._last_bar_time = time.monotonic()
                except ConnectionError as ce:
                    logger.critical(f"ReconnectingFeed: health reconnect failed: {ce}")

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Disconnect, wait with backoff, reconnect. Fires on_reconnect callback."""
        self._reconnect_count += 1
        attempt = 0

        logger.info(
            f"ReconnectingFeed: starting reconnect #{self._reconnect_count}"
        )

        try:
            await self._feed.disconnect()
        except Exception:
            pass

        while attempt < self._max_attempts:
            if self._breaker.is_open:
                if self._on_circuit_open:
                    self._on_circuit_open()
                raise ConnectionError("Circuit breaker OPEN — reconnection aborted")

            wait = exponential_backoff(
                attempt, self._base_delay, self._max_delay, self._jitter
            )
            logger.info(
                f"ReconnectingFeed: reconnect attempt {attempt + 1}/"
                f"{self._max_attempts} in {wait:.1f}s"
            )
            await asyncio.sleep(wait)

            try:
                await self._feed.connect()
                self._breaker.record_success()
                logger.info(
                    f"ReconnectingFeed: reconnected successfully "
                    f"(#{self._reconnect_count})"
                )
                if self._on_reconnect:
                    await self._on_reconnect(self._reconnect_count)
                return

            except Exception as exc:
                self._breaker.record_failure()
                logger.warning(
                    f"ReconnectingFeed: reconnect attempt {attempt + 1} failed: {exc}"
                )
                attempt += 1

        raise ConnectionError(
            f"Reconnection failed after {self._max_attempts} attempts"
        )

    # ------------------------------------------------------------------
    # Circuit breaker control
    # ------------------------------------------------------------------

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker after operator intervention."""
        self._breaker.reset()

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def is_connected(self) -> bool:
        return self._feed.is_connected

    # ------------------------------------------------------------------
    # Delegate status to wrapped feed
    # ------------------------------------------------------------------

    @property
    def status(self):
        return self._feed.status

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ReconnectingFeed":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()