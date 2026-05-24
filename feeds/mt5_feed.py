"""
feeds/mt5_feed.py
-----------------
MetaTrader 5 data feed.

Responsibilities:
  - Connect/disconnect to a running MT5 terminal on this machine
  - Fetch historical bars (warmup)
  - Poll for new closed bars on a configurable interval
  - Normalize all MT5 data into canonical Bar objects with UTC timestamps

MT5 specifics:
  - MT5 timestamps are broker-timezone integers (seconds since epoch).
    We convert them to UTC immediately upon receipt — nowhere else in the
    system should broker-local time appear.
  - MT5 has no push/WebSocket API. We poll. Poll interval = 1s for M1,
    longer for higher timeframes (configurable).
  - The terminal must be open and logged in on the same machine.
  - Thread safety: MT5's Python API is not thread-safe. We run it in an
    executor to keep the event loop unblocked.

Timeframe mapping: MT5 uses integer constants (TIMEFRAME_M1 = 1, etc.).
We accept string keys ("M1", "H1", …) and map internally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from feeds.base_feed import BaseFeed, Bar

# MT5 is imported lazily inside methods so the rest of the system can be
# imported and tested on machines without MT5 installed.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# ---------------------------------------------------------------------------
# Timeframe string → MT5 constant mapping
# ---------------------------------------------------------------------------
_TF_MAP: dict[str, int] = {}

def _build_tf_map() -> None:
    if not MT5_AVAILABLE:
        return
    global _TF_MAP
    _TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,
        "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,
        "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H2":  mt5.TIMEFRAME_H2,
        "H3":  mt5.TIMEFRAME_H3,
        "H4":  mt5.TIMEFRAME_H4,
        "H6":  mt5.TIMEFRAME_H6,
        "H8":  mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


# ---------------------------------------------------------------------------
# How many seconds to wait between polls for each timeframe
# ---------------------------------------------------------------------------
_POLL_INTERVAL: dict[str, float] = {
    "M1": 1.0, "M2": 2.0, "M3": 3.0, "M4": 4.0,
    "M5": 5.0, "M10": 10.0, "M15": 15.0, "M30": 30.0,
    "H1": 30.0, "H4": 60.0, "D1": 120.0,
}


# ---------------------------------------------------------------------------
# Feed implementation
# ---------------------------------------------------------------------------

class MT5Feed(BaseFeed):
    """
    Polls MetaTrader 5 for new closed bars and emits canonical Bar objects.

    Usage (async context manager):
        async with MT5Feed(login=..., password=..., server=...) as feed:
            bars = await feed.fetch_historical("XAUUSD", "H1", 300)
            async for bar in feed.stream(["XAUUSD", "EURUSD"], ["M5", "H1"]):
                await store.write(bar)
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        timeout_ms: int = 10_000,
        poll_jitter: float = 0.1,   # seconds of random jitter to add to polls
    ) -> None:
        super().__init__(feed_id="mt5")
        self._login = login
        self._password = password
        self._server = server
        self._timeout_ms = timeout_ms
        self._poll_jitter = poll_jitter
        # Single-thread executor — MT5 API must run in the same thread
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")
        # Track the last seen bar time per (symbol, timeframe)
        self._last_seen: dict[tuple[str, str], datetime] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not installed. "
                "Run: pip install MetaTrader5"
            )
        _build_tf_map()
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self._connect_sync
        )

    def _connect_sync(self) -> None:
        if not mt5.initialize(
            login=self._login,
            password=self._password,
            server=self._server,
            timeout=self._timeout_ms,
        ):
            err = mt5.last_error()
            raise ConnectionError(f"MT5 initialization failed: {err}")

        info = mt5.account_info()
        if info is None:
            raise ConnectionError("MT5 connected but could not retrieve account info")

        self._status.is_connected = True
        logger.info(
            f"MT5 connected | account={info.login} | "
            f"broker={info.company} | server={info.server} | "
            f"equity={info.equity:.2f} {info.currency}"
        )

    async def disconnect(self) -> None:
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self._disconnect_sync
        )

    def _disconnect_sync(self) -> None:
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._status.is_connected = False
        self._executor.shutdown(wait=False)
        logger.info("MT5 disconnected")

    # ------------------------------------------------------------------
    # Historical fetch
    # ------------------------------------------------------------------

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
    ) -> list[Bar]:
        """
        Fetch the last n_bars *closed* bars.
        Note: MT5 copy_rates_from_pos(0) includes the currently open bar.
              We request n_bars + 1 and drop the last (open) bar.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._fetch_historical_sync,
            symbol, timeframe, n_bars,
        )

    def _fetch_historical_sync(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
    ) -> list[Bar]:
        tf_const = self._resolve_timeframe(timeframe)
        # +1 so we can drop the open (incomplete) bar
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars + 1)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise RuntimeError(
                f"MT5 copy_rates_from_pos failed for {symbol}/{timeframe}: {err}"
            )

        # Drop the last bar (still open)
        rates = rates[:-1]
        now_utc = self.utcnow()

        bars = [self._rate_to_bar(r, symbol, timeframe, now_utc) for r in rates]
        logger.debug(f"Fetched {len(bars)} historical bars | {symbol}/{timeframe}")
        return bars

    # ------------------------------------------------------------------
    # Streaming (poll-based)
    # ------------------------------------------------------------------

    async def stream(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> AsyncIterator[Bar]:
        """
        Polls MT5 continuously. Yields each newly closed bar exactly once.

        The poll interval adapts to the shortest timeframe requested.
        We track the last seen bar timestamp per (symbol, timeframe) and
        only yield bars with a newer event_time.
        """
        if not self._status.is_connected:
            await self.connect()

        # Warm up the last-seen tracker so we don't replay old bars
        await self._initialize_last_seen(symbols, timeframes)

        min_poll = min(
            _POLL_INTERVAL.get(tf, 30.0) for tf in timeframes
        )

        logger.info(
            f"MT5 stream started | symbols={symbols} | "
            f"timeframes={timeframes} | poll={min_poll}s"
        )

        while True:
            poll_start = asyncio.get_event_loop().time()

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        new_bars = await self._poll_new_bars(symbol, timeframe)
                        for bar in new_bars:
                            await self._emit(bar)
                            yield bar
                    except Exception as exc:          # noqa: BLE001
                        self._status.errors += 1
                        self._status.last_error = str(exc)
                        logger.warning(
                            f"MT5 poll error {symbol}/{timeframe}: {exc}"
                        )

            # Sleep for the remainder of the poll interval
            elapsed = asyncio.get_event_loop().time() - poll_start
            sleep_time = max(0.0, min_poll - elapsed)
            await asyncio.sleep(sleep_time)

    async def _poll_new_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> list[Bar]:
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._poll_new_bars_sync,
            symbol, timeframe,
        )

    def _poll_new_bars_sync(
        self,
        symbol: str,
        timeframe: str,
    ) -> list[Bar]:
        """
        Fetch the 2 most recent bars. Return the most recent closed bar
        only if it's newer than what we've seen before.

        We fetch 2 bars because bars[0] = most recent closed, bars[1] = open.
        (MT5 pos=0 is the current open bar, pos=1 is the last closed.)
        """
        tf_const = self._resolve_timeframe(timeframe)
        # Fetch 3: positions 0 (open), 1 (last closed), 2 (second last closed)
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 3)

        if rates is None or len(rates) < 2:
            return []

        now_utc = self.utcnow()
        new_bars: list[Bar] = []
        key = (symbol, timeframe)
        last_seen = self._last_seen.get(key)

        # rates[-1] is the open (current) bar — skip it
        # rates[-2] is the most recently closed bar
        for rate in rates[:-1]:
            bar = self._rate_to_bar(rate, symbol, timeframe, now_utc)
            if last_seen is None or bar.event_time > last_seen:
                new_bars.append(bar)

        if new_bars:
            self._last_seen[key] = max(b.event_time for b in new_bars)

        return new_bars

    async def _initialize_last_seen(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> None:
        """Prime _last_seen so stream() doesn't replay historical bars."""
        for symbol in symbols:
            for timeframe in timeframes:
                bars = await self.fetch_historical(symbol, timeframe, 1)
                if bars:
                    self._last_seen[(symbol, timeframe)] = bars[-1].event_time

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _rate_to_bar(
        self,
        rate,               # numpy void from MT5
        symbol: str,
        timeframe: str,
        publish_time: datetime,
    ) -> Bar:
        """
        Convert a raw MT5 rate record to a canonical Bar.

        MT5 rate fields:
            time        : bar open time (UTC unix seconds)
            open, high, low, close : floats
            tick_volume : volume proxy (actual volume not available for forex)
            spread      : in points
            real_volume : 0 for forex/CFD
        """
        event_time = datetime.fromtimestamp(
            int(rate["time"]), tz=timezone.utc
        )
        # MT5 timestamps the bar by its *open* time.
        # We adjust to close time by adding one timeframe duration.
        event_time = event_time + self._timeframe_duration(timeframe)

        return Bar(
            symbol=symbol,
            timeframe=timeframe,
            event_time=event_time,
            publish_time=publish_time,
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            volume=float(rate["tick_volume"]),
            spread=float(rate["spread"]) if "spread" in rate.dtype.names else 0.0,
            source="mt5",
        )

    @staticmethod
    def _resolve_timeframe(tf_str: str) -> int:
        const = _TF_MAP.get(tf_str.upper())
        if const is None:
            raise ValueError(
                f"Unknown timeframe '{tf_str}'. "
                f"Valid: {list(_TF_MAP.keys())}"
            )
        return const

    @staticmethod
    def _timeframe_duration(tf_str: str) -> timedelta:
        """Return the duration of one bar for a given timeframe string."""
        mapping = {
            "M1": timedelta(minutes=1),
            "M2": timedelta(minutes=2),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
            "W1": timedelta(weeks=1),
            "MN1": timedelta(days=30),
        }
        return mapping.get(tf_str.upper(), timedelta(minutes=1))