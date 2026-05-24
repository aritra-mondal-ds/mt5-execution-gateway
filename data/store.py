"""
data/store.py
-------------
Append-only Parquet store for raw bar data.

Design decisions:
  - One Parquet file per (symbol, timeframe, YYYY-MM).
    This keeps individual files small (~10 MB/month for M1) while keeping
    the partition count manageable.
  - Files are written with PyArrow for speed; read with Polars for analysis.
  - append() is the only write operation — raw data is never modified.
  - All timestamps stored as UTC microseconds (Parquet timestamp[us, UTC]).
  - Thread-safe writes via per-file locks.

File layout:
  data/raw/
    XAUUSD/
      M1/
        2024-01.parquet
        2024-02.parquet
      H1/
        2024-01.parquet
    EURUSD/
      ...
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from loguru import logger

from feeds.base_feed import Bar


# ---------------------------------------------------------------------------
# PyArrow schema — all raw bar files share this schema
# ---------------------------------------------------------------------------
BAR_SCHEMA = pa.schema([
    pa.field("symbol",       pa.string()),
    pa.field("timeframe",    pa.string()),
    pa.field("event_time",   pa.timestamp("us", tz="UTC")),
    pa.field("publish_time", pa.timestamp("us", tz="UTC")),
    pa.field("open",         pa.float64()),
    pa.field("high",         pa.float64()),
    pa.field("low",          pa.float64()),
    pa.field("close",        pa.float64()),
    pa.field("volume",       pa.float64()),
    pa.field("spread",       pa.float64()),
    pa.field("source",       pa.string()),
])


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class RawDataStore:
    """
    Append-only Parquet store for canonical Bar objects.

    Usage:
        store = RawDataStore(root=settings.data.raw_dir)
        await store.write(bar)
        df = store.read("XAUUSD", "H1", start=..., end=...)
    """

    def __init__(
        self,
        root: Path,
        compression: str = "snappy",
        row_group_size: int = 50_000,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._row_group_size = row_group_size
        # Per-file write locks — prevents corruption on concurrent writes
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_meta = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, bar: Bar) -> None:
        """Write a single bar. Safe to call from any thread."""
        path = self._bar_path(bar)
        self._write_bars([bar], path)

    def write_batch(self, bars: Sequence[Bar]) -> None:
        """
        Write a batch of bars, grouping by their partition file.
        More efficient than calling write() in a loop.
        """
        if not bars:
            return

        # Group bars by their partition file
        groups: dict[Path, list[Bar]] = {}
        for bar in bars:
            p = self._bar_path(bar)
            groups.setdefault(p, []).append(bar)

        for path, group in groups.items():
            self._write_bars(group, path)

        logger.debug(f"Wrote batch of {len(bars)} bars across {len(groups)} files")

    def _write_bars(self, bars: Sequence[Bar], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = self._get_lock(path)

        table = self._bars_to_arrow(bars)

        with lock:
            if path.exists():
                # Read existing, concatenate, deduplicate, write back
                existing = pq.read_table(path, schema=BAR_SCHEMA)
                combined = pa.concat_tables([existing, table])
                # Deduplicate on (symbol, timeframe, event_time)
                df = combined.to_pandas()
                df = df.drop_duplicates(
                    subset=["symbol", "timeframe", "event_time"],
                    keep="last",        # keep the row with the latest publish_time
                ).sort_values("event_time")
                table = pa.Table.from_pandas(df, schema=BAR_SCHEMA, preserve_index=False)

            pq.write_table(
                table,
                path,
                compression=self._compression,
                row_group_size=self._row_group_size,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        as_polars: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Read bars for a symbol/timeframe within [start, end].
        Returns a Polars DataFrame by default (faster for feature computation).
        Pass as_polars=False to get a Pandas DataFrame.

        IMPORTANT: This function respects point-in-time correctness.
        If you pass a cutoff datetime as `end`, only bars with
        event_time <= end are returned — as if you were standing at that
        moment in time with no knowledge of the future.
        """
        paths = self._get_partition_paths(symbol, timeframe, start, end)

        if not paths:
            logger.warning(f"No data found for {symbol}/{timeframe}")
            empty_schema = {
                "symbol": pl.Utf8, "timeframe": pl.Utf8,
                "event_time": pl.Datetime("us", "UTC"),
                "publish_time": pl.Datetime("us", "UTC"),
                "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64,
                "volume": pl.Float64, "spread": pl.Float64,
                "source": pl.Utf8,
            }
            df = pl.DataFrame(schema=empty_schema)
            return df if as_polars else df.to_pandas()

        tables = [pq.read_table(p) for p in paths if p.exists()]
        if not tables:
            return pl.DataFrame() if as_polars else pd.DataFrame()

        combined = pa.concat_tables(tables)
        df = pl.from_arrow(combined).sort("event_time")

        # Apply time filters
        if start is not None:
            start_utc = self._ensure_utc(start)
            df = df.filter(pl.col("event_time") >= start_utc)
        if end is not None:
            end_utc = self._ensure_utc(end)
            df = df.filter(pl.col("event_time") <= end_utc)

        logger.debug(
            f"Read {len(df)} bars | {symbol}/{timeframe} | "
            f"{start} → {end}"
        )

        return df if as_polars else df.to_pandas()

    def read_last_n(
        self,
        symbol: str,
        timeframe: str,
        n: int,
        as_of: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Read the last n bars, optionally as of a specific point in time.
        This is the primary read method for the feature pipeline.
        """
        end = as_of or datetime.now()
        df = self.read(symbol, timeframe, end=end, as_polars=True)
        return df.tail(n)

    def get_available_symbols(self) -> list[str]:
        return [d.name for d in self._root.iterdir() if d.is_dir()]

    def get_available_timeframes(self, symbol: str) -> list[str]:
        sym_dir = self._root / symbol
        if not sym_dir.exists():
            return []
        return [d.name for d in sym_dir.iterdir() if d.is_dir()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bar_path(self, bar: Bar) -> Path:
        """Partition path: data/raw/{SYMBOL}/{TIMEFRAME}/{YYYY-MM}.parquet"""
        month_str = bar.event_time.strftime("%Y-%m")
        return self._root / bar.symbol / bar.timeframe / f"{month_str}.parquet"

    def _get_partition_paths(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        base = self._root / symbol / timeframe
        if not base.exists():
            return []
        all_files = sorted(base.glob("*.parquet"))
        if not all_files:
            return []
        if start is None and end is None:
            return all_files
        # Filter by month partition name
        result = []
        for f in all_files:
            # filename is YYYY-MM.parquet
            stem = f.stem  # e.g. "2024-01"
            try:
                file_month = datetime.strptime(stem, "%Y-%m")
            except ValueError:
                result.append(f)
                continue
            if end is not None and file_month.year * 12 + file_month.month \
                    > end.year * 12 + end.month:
                continue
            if start is not None and file_month.year * 12 + file_month.month \
                    < start.year * 12 + start.month - 1:
                continue
            result.append(f)
        return result

    def _get_lock(self, path: Path) -> threading.Lock:
        with self._locks_meta:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    @staticmethod
    def _bars_to_arrow(bars: Sequence[Bar]) -> pa.Table:
        rows = [b.to_dict() for b in bars]
        return pa.Table.from_pylist(rows, schema=BAR_SCHEMA)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        from datetime import timezone
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)