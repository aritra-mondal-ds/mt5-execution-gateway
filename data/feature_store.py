"""
data/feature_store.py
---------------------
Parquet-based cache for computed feature DataFrames.

Why cache features separately from raw bars?
  - Feature computation (EMA-200, ATR, ADX) on 50k bars takes ~2-5 seconds.
    Caching avoids recomputing on every backtest run or system restart.
  - The cache is keyed by (symbol, timeframe, pipeline_version).
    Changing the pipeline config busts the cache automatically.
  - Raw data is immutable. Features are derived and recomputable.
    If in doubt, delete the cache — it rebuilds from raw.

Cache invalidation strategy:
  - We store the pipeline config hash alongside the features.
  - On read, we check: if hash(current_config) != hash(stored_config),
    the cache is stale → recompute.
  - We also check: if the cached data's max(event_time) < store's
    max(event_time), new bars exist → recompute.

File layout:
  data/features/
    XAUUSD/
      H1/
        v1_2024-01.parquet   ← v{pipeline_version}_{YYYY-MM}.parquet
    EURUSD/
      ...
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_hash(config_dict: dict) -> str:
    """Stable MD5 hash of a config dict for cache invalidation."""
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Read/write cache for computed feature DataFrames.

    Usage:
        store = FeatureStore(root=settings.data.features_dir)

        # Write
        store.write("XAUUSD", "H1", feature_df, config_hash="abc123")

        # Read (returns None if cache miss or stale)
        df = store.read("XAUUSD", "H1", config_hash="abc123")
        if df is None:
            df = pipeline.compute(...)  # recompute on miss
            store.write(...)

        # Invalidate one symbol/timeframe
        store.invalidate("XAUUSD", "H1")

        # Invalidate everything
        store.invalidate_all()
    """

    def __init__(
        self,
        root: Path,
        compression: str = "snappy",
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_meta = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        symbol: str,
        timeframe: str,
        df: pl.DataFrame,
        config_hash: str = "default",
    ) -> None:
        """
        Write a feature DataFrame to cache, partitioned by month.
        Overwrites any existing cache for this partition.
        """
        if df is None or len(df) == 0:
            return

        if "event_time" not in df.columns:
            logger.warning("FeatureStore.write: DataFrame has no event_time column — skipping cache")
            return

        # Partition by month (same strategy as raw store)
        df = df.with_columns(
            pl.col("event_time").dt.strftime("%Y-%m").alias("_partition")
        )

        partitions = df["_partition"].unique().to_list()

        for partition in partitions:
            partition_df = df.filter(pl.col("_partition") == partition).drop("_partition")
            path = self._partition_path(symbol, timeframe, partition, config_hash)
            path.parent.mkdir(parents=True, exist_ok=True)

            lock = self._get_lock(path)
            with lock:
                table = partition_df.to_arrow()
                pq.write_table(
                    table,
                    path,
                    compression=self._compression,
                )

        logger.debug(
            f"FeatureStore cached {len(df)} rows | {symbol}/{timeframe} | "
            f"hash={config_hash} | {len(partitions)} partition(s)"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str = "default",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame | None:
        """
        Read cached features.

        Returns:
            pl.DataFrame if cache hit and valid.
            None if cache miss (recompute required).
        """
        paths = self._get_partition_paths(
            symbol, timeframe, config_hash, start, end
        )

        if not paths:
            logger.debug(f"FeatureStore cache miss: {symbol}/{timeframe} hash={config_hash}")
            return None

        try:
            tables = []
            for p in paths:
                if p.exists():
                    tables.append(pq.read_table(p))

            if not tables:
                return None

            combined = pa.concat_tables(tables)
            df = pl.from_arrow(combined).sort("event_time")

            # Apply time filters
            if start is not None:
                df = df.filter(pl.col("event_time") >= _ensure_utc(start))
            if end is not None:
                df = df.filter(pl.col("event_time") <= _ensure_utc(end))

            logger.debug(
                f"FeatureStore cache hit: {symbol}/{timeframe} | "
                f"{len(df)} rows | hash={config_hash}"
            )
            return df

        except Exception as e:
            logger.warning(f"FeatureStore read error ({symbol}/{timeframe}): {e}")
            return None

    def read_latest_row(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str = "default",
    ) -> pl.DataFrame | None:
        """Read only the most recent row from cache. Fast path for live trading."""
        df = self.read(symbol, timeframe, config_hash)
        if df is None or len(df) == 0:
            return None
        return df.tail(1)

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    def is_stale(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str,
        raw_store_max_time: datetime,
    ) -> bool:
        """
        Returns True if the cache needs to be refreshed.

        Cache is stale if:
          - No cached data exists for this hash
          - The cached max(event_time) < raw store's max(event_time)
        """
        df = self.read(symbol, timeframe, config_hash)
        if df is None or len(df) == 0:
            return True

        cached_max = df["event_time"].max()
        if cached_max is None:
            return True

        # Compare as timezone-aware
        if cached_max.tzinfo is None:
            cached_max = cached_max.replace(tzinfo=timezone.utc)

        return cached_max < _ensure_utc(raw_store_max_time)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, symbol: str, timeframe: str) -> int:
        """
        Delete all cached feature files for a symbol/timeframe.
        Returns number of files deleted.
        """
        base = self._root / symbol / timeframe
        if not base.exists():
            return 0

        deleted = 0
        for f in base.glob("*.parquet"):
            f.unlink()
            deleted += 1

        logger.info(f"FeatureStore invalidated {deleted} files for {symbol}/{timeframe}")
        return deleted

    def invalidate_all(self) -> int:
        """Delete all cached feature files. Returns total files deleted."""
        deleted = 0
        for f in self._root.rglob("*.parquet"):
            f.unlink()
            deleted += 1
        logger.info(f"FeatureStore: invalidated all ({deleted} files)")
        return deleted

    def get_cache_info(self) -> list[dict]:
        """Return metadata about what's currently cached."""
        info = []
        for symbol_dir in self._root.iterdir():
            if not symbol_dir.is_dir():
                continue
            for tf_dir in symbol_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                files = list(tf_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files)
                info.append({
                    "symbol": symbol_dir.name,
                    "timeframe": tf_dir.name,
                    "partitions": len(files),
                    "size_mb": round(total_size / 1024 / 1024, 2),
                })
        return info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _partition_path(
        self,
        symbol: str,
        timeframe: str,
        partition: str,   # "YYYY-MM"
        config_hash: str,
    ) -> Path:
        """v{hash}_{YYYY-MM}.parquet"""
        return self._root / symbol / timeframe / f"v{config_hash}_{partition}.parquet"

    def _get_partition_paths(
        self,
        symbol: str,
        timeframe: str,
        config_hash: str,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        base = self._root / symbol / timeframe
        if not base.exists():
            return []

        prefix = f"v{config_hash}_"
        all_files = sorted(f for f in base.glob(f"{prefix}*.parquet"))

        if not all_files:
            return []

        if start is None and end is None:
            return all_files

        result = []
        for f in all_files:
            # filename: v{hash}_{YYYY-MM}.parquet
            stem = f.stem  # e.g. "vabc123_2024-01"
            partition = stem.split("_", 1)[-1]   # "2024-01"
            try:
                file_month = datetime.strptime(partition, "%Y-%m")
            except ValueError:
                result.append(f)
                continue

            ym = file_month.year * 12 + file_month.month
            if end is not None:
                end_ym = end.year * 12 + end.month
                if ym > end_ym:
                    continue
            if start is not None:
                start_ym = start.year * 12 + start.month - 1
                if ym < start_ym:
                    continue
            result.append(f)

        return result

    def _get_lock(self, path: Path) -> threading.Lock:
        with self._locks_meta:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)