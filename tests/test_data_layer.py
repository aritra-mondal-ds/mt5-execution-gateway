"""
tests/test_data_layer.py
------------------------
Tests for the complete data layer:
    - Bar dataclass correctness and invariants
    - BaseFeed interface contract
    - MT5Feed (without a live terminal — uses a mock)
    - RawDataStore read/write/deduplication

Run with:
    pytest tests/test_data_layer.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

import pytest
import polars as pl

from feeds.base_feed import Bar, BaseFeed, FeedStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_bar(
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    event_time: datetime | None = None,
    close: float = 2000.0,
) -> Bar:
    now = datetime.now(tz=timezone.utc)
    et = event_time or now
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        event_time=et,
        publish_time=now,
        open=1998.0,
        high=2005.0,
        low=1995.0,
        close=close,
        volume=12345.0,
        spread=2.0,
        source="test",
    )


@pytest.fixture
def tmp_store(tmp_path: Path):
    from data.store import RawDataStore
    return RawDataStore(root=tmp_path / "raw")


# ---------------------------------------------------------------------------
# Bar tests
# ---------------------------------------------------------------------------

class TestBar:
    def test_basic_construction(self):
        bar = make_bar()
        assert bar.symbol == "XAUUSD"
        assert bar.close == 2000.0
        assert bar.is_bullish  # close >= open

    def test_utc_required_on_event_time(self):
        with pytest.raises(AssertionError, match="timezone-aware"):
            Bar(
                symbol="XAUUSD", timeframe="H1",
                event_time=datetime(2024, 1, 1),       # naive — must fail
                publish_time=datetime.now(tz=timezone.utc),
                open=100, high=110, low=90, close=105, volume=1,
            )

    def test_utc_required_on_publish_time(self):
        with pytest.raises(AssertionError, match="timezone-aware"):
            Bar(
                symbol="XAUUSD", timeframe="H1",
                event_time=datetime.now(tz=timezone.utc),
                publish_time=datetime(2024, 1, 1),     # naive — must fail
                open=100, high=110, low=90, close=105, volume=1,
            )

    def test_high_must_be_gte_low(self):
        with pytest.raises(AssertionError):
            Bar(
                symbol="XAUUSD", timeframe="H1",
                event_time=datetime.now(tz=timezone.utc),
                publish_time=datetime.now(tz=timezone.utc),
                open=100, high=80, low=90, close=95, volume=1,  # high < low
            )

    def test_body_size(self):
        bar = make_bar()
        assert bar.body_size == abs(bar.close - bar.open)

    def test_range_size(self):
        bar = make_bar()
        assert bar.range_size == bar.high - bar.low

    def test_to_dict_keys(self):
        bar = make_bar()
        d = bar.to_dict()
        expected_keys = {
            "symbol", "timeframe", "event_time", "publish_time",
            "open", "high", "low", "close", "volume", "spread", "source",
        }
        assert set(d.keys()) == expected_keys

    def test_frozen_immutable(self):
        bar = make_bar()
        with pytest.raises((AttributeError, TypeError)):
            bar.close = 9999.0  # type: ignore


# ---------------------------------------------------------------------------
# BaseFeed interface contract
# ---------------------------------------------------------------------------

class ConcreteTestFeed(BaseFeed):
    """Minimal concrete subclass for testing the base contract."""

    def __init__(self):
        super().__init__(feed_id="test_feed")
        self.emitted: list[Bar] = []

    async def connect(self) -> None:
        self._status.is_connected = True

    async def disconnect(self) -> None:
        self._status.is_connected = False

    async def fetch_historical(self, symbol, timeframe, n_bars) -> list[Bar]:
        return [make_bar(symbol=symbol, timeframe=timeframe) for _ in range(n_bars)]

    async def stream(self, symbols, timeframes):
        for symbol in symbols:
            for timeframe in timeframes:
                bar = make_bar(symbol=symbol, timeframe=timeframe)
                yield bar


class TestBaseFeed:
    def test_initial_status(self):
        feed = ConcreteTestFeed()
        assert not feed.is_connected
        assert feed.status.bars_received == 0

    def test_context_manager(self):
        async def run():
            async with ConcreteTestFeed() as feed:
                assert feed.is_connected
            assert not feed.is_connected
        asyncio.run(run())

    def test_callback_registration_and_emit(self):
        received = []

        async def run():
            feed = ConcreteTestFeed()
            feed.on_bar(lambda b: received.append(b) or asyncio.coroutine(lambda: None)())

            # Manually emit a bar
            bar = make_bar()
            # Patch emit to collect
            async def collect(b):
                received.append(b)
            feed._bar_callbacks = [collect]
            await feed._emit(bar)

        asyncio.run(run())
        assert len(received) == 1

    def test_status_updated_on_emit(self):
        async def run():
            feed = ConcreteTestFeed()
            bar = make_bar()
            await feed._emit(bar)
            assert feed.status.bars_received == 1
            assert feed.status.last_bar_time is not None

        asyncio.run(run())

    def test_fetch_historical_returns_correct_count(self):
        async def run():
            feed = ConcreteTestFeed()
            bars = await feed.fetch_historical("XAUUSD", "H1", 10)
            assert len(bars) == 10
            assert all(isinstance(b, Bar) for b in bars)

        asyncio.run(run())

    def test_stream_yields_bars(self):
        async def run():
            feed = ConcreteTestFeed()
            collected = []
            async for bar in feed.stream(["XAUUSD"], ["H1", "M5"]):
                collected.append(bar)
            assert len(collected) == 2
            assert all(isinstance(b, Bar) for b in collected)

        asyncio.run(run())

    def test_utcnow_is_aware(self):
        feed = ConcreteTestFeed()
        now = feed.utcnow()
        assert now.tzinfo is not None

    def test_to_utc_converts_naive(self):
        feed = ConcreteTestFeed()
        naive = datetime(2024, 6, 1, 12, 0, 0)
        aware = feed.to_utc(naive)
        assert aware.tzinfo is not None


# ---------------------------------------------------------------------------
# RawDataStore tests
# ---------------------------------------------------------------------------

class TestRawDataStore:
    def test_write_and_read_single_bar(self, tmp_store):
        bar = make_bar()
        tmp_store.write(bar)
        df = tmp_store.read("XAUUSD", "H1", as_polars=True)
        assert len(df) == 1
        assert df["symbol"][0] == "XAUUSD"
        assert df["close"][0] == bar.close

    def test_write_batch(self, tmp_store):
        base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(
                event_time=base + timedelta(days=i),
                close=2000.0,   # keep valid with high=2005.0
            )
            for i in range(20)
        ]
        tmp_store.write_batch(bars)
        df = tmp_store.read("XAUUSD", "H1")
        assert len(df) == 20

    # def test_write_batch(self, tmp_store):
    #     bars = [
    #         make_bar(
    #             event_time=datetime(2024, 1, i + 1, 10, 0, tzinfo=timezone.utc),
    #             close=2000.0 + i,
    #         )
    #         for i in range(20)
    #     ]
    #     tmp_store.write_batch(bars)
    #     df = tmp_store.read("XAUUSD", "H1")
    #     assert len(df) == 20

    def test_deduplication(self, tmp_store):
        """Writing the same bar twice must result in one row."""
        bar = make_bar(event_time=datetime(2024, 3, 15, 9, 0, tzinfo=timezone.utc))
        tmp_store.write(bar)
        tmp_store.write(bar)
        df = tmp_store.read("XAUUSD", "H1")
        assert len(df) == 1

    def test_time_range_filter(self, tmp_store):
        bars = [
            make_bar(event_time=datetime(2024, 1, d, 10, 0, tzinfo=timezone.utc))
            for d in range(1, 11)
        ]
        tmp_store.write_batch(bars)

        start = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 7, 10, 0, tzinfo=timezone.utc)
        df = tmp_store.read("XAUUSD", "H1", start=start, end=end)

        assert len(df) == 4  # Jan 4, 5, 6, 7

    # def test_time_range_filter(self, tmp_store):
    #     bars = [
    #         make_bar(event_time=datetime(2024, 1, d, 10, 0, tzinfo=timezone.utc))
    #         for d in range(1, 11)  # Jan 1–10
    #     ]
    #     tmp_store.write_batch(bars)

    #     start = datetime(2024, 1, 4, tzinfo=timezone.utc)
    #     end = datetime(2024, 1, 7, tzinfo=timezone.utc)
    #     df = tmp_store.read("XAUUSD", "H1", start=start, end=end)
    #     assert len(df) == 4  # Jan 4, 5, 6, 7

    def test_read_last_n(self, tmp_store):
        base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(event_time=base + timedelta(days=i))
            for i in range(50)
        ]
        # bars = [
        #     make_bar(event_time=datetime(2024, 1, i + 1, 10, 0, tzinfo=timezone.utc))
        #     for i in range(50)
        # ]
        tmp_store.write_batch(bars)
        df = tmp_store.read_last_n("XAUUSD", "H1", n=10)
        assert len(df) == 10
        # Should be the 10 most recent
        assert df["event_time"][-1] == bars[-1].event_time

    def test_read_empty_returns_empty_df(self, tmp_store):
        df = tmp_store.read("NONEXISTENT", "M1")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_returns_pandas_when_requested(self, tmp_store):
        import pandas as pd
        bar = make_bar()
        tmp_store.write(bar)
        df = tmp_store.read("XAUUSD", "H1", as_polars=False)
        assert isinstance(df, pd.DataFrame)

    def test_bars_sorted_ascending(self, tmp_store):
        """Store must always return bars oldest-first."""
        # Write in reverse order
        bars = [
            make_bar(event_time=datetime(2024, 1, d, 10, 0, tzinfo=timezone.utc))
            for d in [5, 3, 1, 4, 2]
        ]
        tmp_store.write_batch(bars)
        df = tmp_store.read("XAUUSD", "H1")
        times = df["event_time"].to_list()
        assert times == sorted(times)

    def test_multi_symbol_isolation(self, tmp_store):
        """Bars for different symbols must not bleed into each other."""
        tmp_store.write(make_bar(symbol="XAUUSD"))
        tmp_store.write(make_bar(symbol="EURUSD"))
        df_gold = tmp_store.read("XAUUSD", "H1")
        df_eur = tmp_store.read("EURUSD", "H1")
        assert len(df_gold) == 1
        assert len(df_eur) == 1
        assert df_gold["symbol"][0] == "XAUUSD"
        assert df_eur["symbol"][0] == "EURUSD"

    def test_get_available_symbols(self, tmp_store):
        tmp_store.write(make_bar(symbol="XAUUSD"))
        tmp_store.write(make_bar(symbol="BTCUSDT"))
        symbols = tmp_store.get_available_symbols()
        assert set(symbols) == {"XAUUSD", "BTCUSDT"}

    def test_partition_by_month(self, tmp_store):
        """Bars in different months should land in different files."""
        bar_jan = make_bar(event_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc))
        bar_feb = make_bar(event_time=datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc))
        tmp_store.write(bar_jan)
        tmp_store.write(bar_feb)

        jan_file = tmp_store._root / "XAUUSD" / "H1" / "2024-01.parquet"
        feb_file = tmp_store._root / "XAUUSD" / "H1" / "2024-02.parquet"
        assert jan_file.exists()
        assert feb_file.exists()