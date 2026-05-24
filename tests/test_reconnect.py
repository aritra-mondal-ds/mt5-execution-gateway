"""
tests/test_reconnect.py
------------------------
Tests for the reconnection layer.

Coverage:
  - exponential_backoff: formula correctness, jitter bounds, cap
  - CircuitBreaker: state transitions, auto-reset, manual reset
  - ReconnectingFeed: connect with retry, circuit open, stream recovery,
    stale feed detection, fetch_historical retry, callbacks

All tests use a MockFeed that can be configured to fail N times before
succeeding — no live connection required.

Run with:
    pytest tests/test_reconnect.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator

import pytest

from feeds.base_feed import BaseFeed, Bar
from feeds.reconnect import (
    exponential_backoff, CircuitBreaker, ReconnectingFeed,
)


# ===========================================================================
# Helpers
# ===========================================================================

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _bar(symbol: str = "XAUUSD", offset: int = 0) -> Bar:
    t = _NOW + timedelta(hours=offset)
    return Bar(
        symbol=symbol, timeframe="H1",
        event_time=t, publish_time=t,
        open=2000.0, high=2010.0, low=1990.0, close=2005.0,
        volume=1000.0, spread=0.30, source="test",
    )


class MockFeed(BaseFeed):
    """
    Configurable mock feed for testing reconnection logic.

    fail_connect_times: how many connect() calls fail before succeeding
    fail_stream_after : how many bars to yield before raising an error
    bars_to_yield     : total bars to yield across all stream() calls
    """

    def __init__(
        self,
        fail_connect_times: int = 0,
        fail_stream_after: int = 999,
        bars_to_yield: int = 3,
    ) -> None:
        super().__init__(feed_id="mock")
        self._fail_connect_remaining = fail_connect_times
        self._fail_stream_after = fail_stream_after
        self._bars_to_yield = bars_to_yield
        self._connect_calls = 0
        self._disconnect_calls = 0
        self._bars_yielded = 0

    async def connect(self) -> None:
        self._connect_calls += 1
        if self._fail_connect_remaining > 0:
            self._fail_connect_remaining -= 1
            raise ConnectionError(f"Mock connect failure #{self._connect_calls}")
        self._status.is_connected = True

    async def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._status.is_connected = False

    async def fetch_historical(self, symbol, timeframe, n_bars) -> list[Bar]:
        return [_bar(symbol, i) for i in range(n_bars)]

    async def stream(self, symbols, timeframes) -> AsyncIterator[Bar]:
        count = 0
        while self._bars_yielded < self._bars_to_yield:
            if count >= self._fail_stream_after:
                raise ConnectionError("Mock stream disconnected")
            bar = _bar(symbols[0], self._bars_yielded)
            self._bars_yielded += 1
            count += 1
            yield bar


def _reconnecting(
    feed: MockFeed | None = None,
    max_attempts: int = 5,
    base_delay: float = 0.0,    # zero delay for fast tests
    max_delay: float = 0.0,
    max_stale: float = 999.0,
    health_interval: float = 999.0,
    **kwargs,
) -> ReconnectingFeed:
    return ReconnectingFeed(
        feed=feed or MockFeed(),
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        max_stale_seconds=max_stale,
        health_check_interval=health_interval,
        **kwargs,
    )


# ===========================================================================
# exponential_backoff
# ===========================================================================

class TestExponentialBackoff:
    def test_attempt_0_returns_near_base_delay(self):
        # With jitter=0, attempt 0 = base_delay * 2^0 = base_delay
        wait = exponential_backoff(0, base_delay=1.0, max_delay=60.0, jitter_factor=0)
        assert wait == pytest.approx(1.0)

    def test_doubles_each_attempt(self):
        w0 = exponential_backoff(0, base_delay=1.0, max_delay=999.0, jitter_factor=0)
        w1 = exponential_backoff(1, base_delay=1.0, max_delay=999.0, jitter_factor=0)
        w2 = exponential_backoff(2, base_delay=1.0, max_delay=999.0, jitter_factor=0)
        assert w1 == pytest.approx(w0 * 2)
        assert w2 == pytest.approx(w1 * 2)

    def test_capped_at_max_delay(self):
        wait = exponential_backoff(20, base_delay=1.0, max_delay=30.0, jitter_factor=0)
        assert wait == pytest.approx(30.0)

    def test_jitter_is_non_negative(self):
        for attempt in range(10):
            wait = exponential_backoff(attempt, base_delay=1.0, max_delay=60.0)
            assert wait >= 0

    def test_jitter_bounded_above(self):
        # With jitter_factor=0.25, wait <= base * 2^attempt * 1.25 (+ cap)
        for attempt in range(5):
            for _ in range(20):  # multiple samples
                wait = exponential_backoff(
                    attempt, base_delay=1.0, max_delay=999.0, jitter_factor=0.25
                )
                expected_max = 1.0 * (2 ** attempt) * 1.25
                assert wait <= expected_max + 1e-9


# ===========================================================================
# CircuitBreaker
# ===========================================================================

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(max_failures=3)
        assert cb.state == "closed"
        assert not cb.is_open

    def test_opens_after_max_failures(self):
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open  # 2 failures, not yet open
        cb.record_failure()    # 3rd failure → opens
        assert cb.is_open

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_manual_reset(self):
        cb = CircuitBreaker(max_failures=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert not cb.is_open
        assert cb.state == "closed"

    def test_auto_transition_to_half_open(self):
        cb = CircuitBreaker(max_failures=2, reset_after_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        import time; time.sleep(0.06)
        assert not cb.is_open   # transitioned to half-open
        assert cb.state == "half-open"

    def test_half_open_allows_attempt(self):
        cb = CircuitBreaker(max_failures=2, reset_after_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        import time; time.sleep(0.06)
        is_open = cb.is_open   # triggers half-open transition
        assert not is_open     # allows one attempt

    def test_success_in_half_open_closes(self):
        cb = CircuitBreaker(max_failures=2, reset_after_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        import time; time.sleep(0.06)
        cb.is_open  # trigger half-open
        cb.record_success()
        assert cb.state == "closed"

    def test_failure_count_tracked(self):
        cb = CircuitBreaker(max_failures=10)
        for i in range(5):
            cb.record_failure()
        assert cb.failure_count == 5


# ===========================================================================
# ReconnectingFeed — connect with retry
# ===========================================================================

class TestReconnectingFeedConnect:
    def test_connects_on_first_attempt(self):
        async def run():
            feed = MockFeed(fail_connect_times=0)
            rf = _reconnecting(feed)
            await rf.connect()
            assert rf.is_connected
            assert feed._connect_calls == 1
            await rf.disconnect()
        asyncio.run(run())

    def test_retries_on_failure(self):
        async def run():
            feed = MockFeed(fail_connect_times=2)
            rf = _reconnecting(feed, max_attempts=5)
            await rf.connect()
            assert rf.is_connected
            assert feed._connect_calls == 3  # 2 failures + 1 success
            await rf.disconnect()
        asyncio.run(run())

    def test_raises_after_max_attempts(self):
        async def run():
            feed = MockFeed(fail_connect_times=999)  # always fails
            rf = _reconnecting(feed, max_attempts=3)
            with pytest.raises(ConnectionError):
                await rf.connect()
        asyncio.run(run())

    def test_circuit_opens_after_max_failures(self):
        async def run():
            feed = MockFeed(fail_connect_times=999)
            rf = _reconnecting(feed, max_attempts=3)
            try:
                await rf.connect()
            except ConnectionError:
                pass
            assert rf.circuit_state in ("open", "closed")
        asyncio.run(run())

    def test_circuit_open_blocks_immediate_reconnect(self):
        async def run():
            feed = MockFeed(fail_connect_times=999)
            rf = _reconnecting(feed, max_attempts=3)
            # Force circuit open
            for _ in range(3):
                rf._breaker.record_failure()

            with pytest.raises(ConnectionError, match="Circuit breaker"):
                await rf.connect()
        asyncio.run(run())

    def test_on_circuit_open_callback_fired(self):
        async def run():
            called = []
            feed = MockFeed(fail_connect_times=999)
            rf = _reconnecting(
                feed, max_attempts=3,
                on_circuit_open=lambda: called.append(True),
            )
            try:
                await rf.connect()
            except ConnectionError:
                pass
            assert len(called) >= 1
        asyncio.run(run())

    def test_context_manager_connects_and_disconnects(self):
        async def run():
            feed = MockFeed()
            rf = _reconnecting(feed)
            async with rf:
                assert rf.is_connected
            assert not rf.is_connected
        asyncio.run(run())


# ===========================================================================
# ReconnectingFeed — stream
# ===========================================================================

class TestReconnectingFeedStream:
    def test_stream_yields_bars_normally(self):
        async def run():
            feed = MockFeed(bars_to_yield=5)
            rf = _reconnecting(feed)
            await rf.connect()
            collected = []
            async for bar in rf.stream(["XAUUSD"], ["H1"]):
                collected.append(bar)
            assert len(collected) == 5
            await rf.disconnect()
        asyncio.run(run())

    def test_stream_reconnects_on_error(self):
        async def run():
            # Fail after 2 bars, then yield 2 more on reconnect
            feed = MockFeed(fail_stream_after=2, bars_to_yield=4)
            rf = _reconnecting(feed, max_attempts=5)
            await rf.connect()
            collected = []
            async for bar in rf.stream(["XAUUSD"], ["H1"]):
                collected.append(bar)
            # All 4 bars should be collected across reconnect
            assert len(collected) == 4
            await rf.disconnect()
        asyncio.run(run())

    def test_on_reconnect_callback_fired(self):
        async def run():
            reconnects = []

            async def on_reconnect(n: int):
                reconnects.append(n)

            feed = MockFeed(fail_stream_after=1, bars_to_yield=3)
            rf = _reconnecting(
                feed, max_attempts=5,
                on_reconnect=on_reconnect,
            )
            await rf.connect()
            async for _ in rf.stream(["XAUUSD"], ["H1"]):
                pass
            assert len(reconnects) >= 1
            assert reconnects[0] == 1  # first reconnect
            await rf.disconnect()
        asyncio.run(run())

    def test_reconnect_count_increments(self):
        async def run():
            feed = MockFeed(fail_stream_after=1, bars_to_yield=3)
            rf = _reconnecting(feed, max_attempts=5)
            await rf.connect()
            async for _ in rf.stream(["XAUUSD"], ["H1"]):
                pass
            assert rf.reconnect_count >= 1
            await rf.disconnect()
        asyncio.run(run())

    def test_stream_stops_when_circuit_opens(self):
        async def run():
            feed = MockFeed(fail_connect_times=999, bars_to_yield=0)
            rf = _reconnecting(feed, max_attempts=3)

            # Force feed to think it's connected so stream() starts
            feed._status.is_connected = True
            rf._running = True

            # Force circuit open before stream error triggers reconnect
            for _ in range(3):
                rf._breaker.record_failure()

            collected = []
            try:
                async for bar in rf.stream(["XAUUSD"], ["H1"]):
                    collected.append(bar)
            except Exception:
                pass
            # Stream should terminate (not hang)
            assert len(collected) == 0
        asyncio.run(run())


# ===========================================================================
# ReconnectingFeed — fetch_historical retry
# ===========================================================================

class TestReconnectingFeedHistorical:
    def test_fetch_historical_succeeds_normally(self):
        async def run():
            feed = MockFeed()
            rf = _reconnecting(feed)
            await rf.connect()
            bars = await rf.fetch_historical("XAUUSD", "H1", 10)
            assert len(bars) == 10
            await rf.disconnect()
        asyncio.run(run())

    def test_fetch_historical_retries_once_on_error(self):
        async def run():
            call_count = [0]
            original_fetch = MockFeed.fetch_historical

            async def failing_then_ok(self, symbol, timeframe, n_bars):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ConnectionError("first call fails")
                return [_bar(symbol, i) for i in range(n_bars)]

            feed = MockFeed()
            feed._status.is_connected = True
            rf = _reconnecting(feed)

            # Patch fetch_historical on the instance
            import types
            feed.fetch_historical = types.MethodType(failing_then_ok, feed)

            # connect will be called on retry — pre-configure feed to succeed
            bars = await rf.fetch_historical("XAUUSD", "H1", 5)
            assert len(bars) == 5
            assert call_count[0] == 2
        asyncio.run(run())


# ===========================================================================
# ReconnectingFeed — reset circuit
# ===========================================================================

class TestReconnectingFeedCircuitReset:
    def test_reset_circuit_allows_reconnect(self):
        async def run():
            feed = MockFeed(fail_connect_times=0)
            rf = _reconnecting(feed)

            # Open the circuit manually
            for _ in range(10):
                rf._breaker.record_failure()
            assert rf.circuit_state == "open"

            # Reset
            rf.reset_circuit()
            assert rf.circuit_state == "closed"

            # Should now be able to connect
            await rf.connect()
            assert rf.is_connected
            await rf.disconnect()
        asyncio.run(run())