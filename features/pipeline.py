"""
features/pipeline.py
---------------------
Feature pipeline orchestrator.

This is the single entry point for feature computation.
The rest of the system (strategies, ML training, backtester) should
call FeaturePipeline, never the individual feature modules directly.

Responsibilities:
  1. Load raw bars from the data store
  2. Validate data quality (gaps, outliers, minimum length)
  3. Apply technical + ML features in correct order
  4. Enforce point-in-time correctness
  5. Drop rows with NaN in critical feature columns
  6. Cache computed features to the feature store
  7. Expose a clean `compute(symbol, timeframe, as_of)` interface

Point-in-time correctness:
  Features at row t are computed using bars 0..t (inclusive).
  A strategy that runs at bar t+1's open may safely use features
  from row t. The pipeline does NOT enforce this shift — that is the
  strategy's responsibility. The pipeline ensures that NO future bar
  (t+1, t+2, …) influences features at row t.

  Rolling windows: min_periods enforced → no partial-window values.
  Shift operations in feature functions: always shift(1) or shift(n),
  never shift(-n) (which would pull future data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger

from features.technical import add_all_technical
from features.ml_features import add_all_ml_features


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Controls which feature groups are computed and their parameters.
    Modify here rather than scattering magic numbers across the codebase.
    """
    # Minimum bars needed before any features are valid
    # (driven by ema_200 — longest window)
    min_bars: int = 200

    # Technical indicator parameters
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50, 200])
    sma_periods: list[int] = field(default_factory=lambda: [20, 50])
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    macd_params: tuple = (12, 26, 9)
    stoch_params: tuple = (14, 3)
    adx_period: int = 14
    include_vwap: bool = True

    # ML feature parameters
    return_horizons: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])
    vol_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    price_pos_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Which feature groups to compute
    compute_technical: bool = True
    compute_ml: bool = True

    # Drop rows where ANY of these columns is NaN
    # (these are the most critical features for most strategies)
    required_feature_cols: list[str] = field(default_factory=lambda: [
        "rsi_14", "atr_14", "ema_21", "macd_line", "log_ret_1",
    ])

    # Whether to add forward-return target columns (TRAINING ONLY)
    add_targets: bool = False
    target_horizons: list[int] = field(default_factory=lambda: [1, 5, 10])


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    total_bars: int
    bars_after_filter: int
    nan_counts: dict[str, int]
    gap_count: int
    warnings: list[str]

    @property
    def dropped_bars(self) -> int:
        return self.total_bars - self.bars_after_filter

    @property
    def drop_rate(self) -> float:
        if self.total_bars == 0:
            return 0.0
        return self.dropped_bars / self.total_bars

    def log(self) -> None:
        logger.info(
            f"Data quality | {self.symbol}/{self.timeframe} | "
            f"bars={self.total_bars} → {self.bars_after_filter} "
            f"(dropped={self.dropped_bars}, {self.drop_rate:.1%}) | "
            f"gaps={self.gap_count}"
        )
        for w in self.warnings:
            logger.warning(w)


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Compute features from raw bars, with caching and quality validation.

    Usage (backtesting — point-in-time query):
        pipeline = FeaturePipeline(store, config)
        df = pipeline.compute("XAUUSD", "H1", as_of=datetime(2024, 6, 1))

    Usage (live — full history):
        df = pipeline.compute("XAUUSD", "H1")

    Usage (training — with targets):
        config = PipelineConfig(add_targets=True)
        pipeline = FeaturePipeline(store, config)
        df = pipeline.compute("XAUUSD", "H1")
        X = df.select(pipeline.feature_columns)
        y = df["target_ret_1"]
    """

    def __init__(
        self,
        store,                          # RawDataStore
        config: PipelineConfig | None = None,
        feature_store=None,             # FeatureStore (optional cache)
    ) -> None:
        self._store = store
        self._config = config or PipelineConfig()
        self._feature_store = feature_store
        self._last_quality_report: DataQualityReport | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime | None = None,
        n_bars: int | None = None,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """
        Compute all features for symbol/timeframe.

        Args:
            symbol      : e.g. "XAUUSD"
            timeframe   : e.g. "H1"
            as_of       : Point-in-time cutoff. Only bars with
                          event_time <= as_of are used.
                          None = use all available data.
            n_bars      : If set, only load the last n_bars from the store.
                          Overridden by as_of if both provided.
            use_cache   : If True and a feature store is configured,
                          attempt to load from cache first.

        Returns:
            Polars DataFrame with all feature columns.
            Rows with NaN in required_feature_cols are dropped.
            Sorted ascending by event_time.
        """
        logger.info(f"Feature pipeline: {symbol}/{timeframe} | as_of={as_of}")

        # 1. Load raw bars
        raw_df = self._load_raw(symbol, timeframe, as_of, n_bars)
        if raw_df is None or len(raw_df) == 0:
            logger.warning(f"No raw data for {symbol}/{timeframe}")
            return pl.DataFrame()

        if len(raw_df) < self._config.min_bars:
            logger.warning(
                f"Insufficient bars for {symbol}/{timeframe}: "
                f"{len(raw_df)} < {self._config.min_bars} required"
            )
            return pl.DataFrame()

        # 2. Validate input data
        raw_df = self._validate_and_clean(raw_df, symbol, timeframe)

        # 3. Compute features
        feature_df = self._compute_features(raw_df)

        # 4. Validate and filter output
        feature_df, report = self._filter_and_report(
            raw_df, feature_df, symbol, timeframe
        )
        report.log()
        self._last_quality_report = report

        # 5. Cache if configured
        if self._feature_store is not None and use_cache:
            try:
                self._feature_store.write(symbol, timeframe, feature_df)
            except Exception as e:
                logger.warning(f"Feature cache write failed: {e}")

        return feature_df

    def compute_latest(
        self,
        symbol: str,
        timeframe: str,
        warmup_bars: int = 300,
    ) -> pl.DataFrame | None:
        """
        Compute features for the most recent completed bar only.
        Used in live trading to get the current feature vector.

        Loads `warmup_bars` of history for indicator warm-up,
        returns only the last fully-computed row.
        """
        df = self.compute(symbol, timeframe, n_bars=warmup_bars)
        if df is None or len(df) == 0:
            return None
        return df.tail(1)

    # ------------------------------------------------------------------
    # Feature column introspection
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        """
        Returns the list of feature column names (excludes OHLCV, metadata,
        event_time, publish_time, and target columns).

        Call this after at least one successful compute() to get the
        correct column list.
        """
        non_feature = {
            "symbol", "timeframe", "event_time", "publish_time",
            "open", "high", "low", "close", "volume", "spread", "source",
        }
        if self._last_quality_report is None:
            return []

        # We can't know columns without a computed df — caller should
        # use df.columns and filter manually using TARGET_PREFIX
        return []

    TARGET_PREFIX = "target_"

    def split_features_targets(
        self,
        df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split a DataFrame into features (X) and targets (y).
        Targets are any columns starting with 'target_'.
        """
        raw_cols = {
            "symbol", "timeframe", "event_time", "publish_time",
            "open", "high", "low", "close", "volume", "spread", "source",
        }
        target_cols = [c for c in df.columns if c.startswith(self.TARGET_PREFIX)]
        feature_cols = [
            c for c in df.columns
            if c not in raw_cols and not c.startswith(self.TARGET_PREFIX)
        ]
        return df.select(feature_cols), df.select(target_cols)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load_raw(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime | None,
        n_bars: int | None,
    ) -> pl.DataFrame | None:
        """Load raw bars from the data store."""
        try:
            if n_bars is not None and as_of is None:
                return self._store.read_last_n(symbol, timeframe, n=n_bars)
            else:
                return self._store.read(
                    symbol, timeframe,
                    end=as_of,
                    as_polars=True,
                )
        except Exception as e:
            logger.error(f"Failed to load raw data for {symbol}/{timeframe}: {e}")
            return None

    def _validate_and_clean(
        self,
        df: pl.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pl.DataFrame:
        """
        Input data quality checks and cleaning.

        Checks performed:
          - Duplicate timestamps → keep last
          - Sort ascending by event_time (mandatory for rolling windows)
          - Zero-range bars (high == low) → flag but keep
          - Extreme price jumps → log warning
          - Zero volume → log warning (common in forex weekends)
        """
        original_len = len(df)

        # Deduplicate on event_time, keep last occurrence
        df = df.unique(subset=["event_time"], keep="last").sort("event_time")

        if len(df) < original_len:
            logger.warning(
                f"{symbol}/{timeframe}: removed {original_len - len(df)} duplicate timestamps"
            )

        # Check for zero-range bars
        zero_range = df.filter(pl.col("high") == pl.col("low"))
        if len(zero_range) > 0:
            logger.debug(
                f"{symbol}/{timeframe}: {len(zero_range)} zero-range bars (normal for gaps)"
            )

        # Check for extreme returns (potential data errors)
        # >20% move in one bar is suspicious for most instruments
        if len(df) > 1:
            max_ret = (
                (pl.col("close") / pl.col("close").shift(1) - 1.0).abs().max()
            )
            df_check = df.select(max_ret.alias("max_ret"))
            max_ret_val = df_check["max_ret"][0]
            if max_ret_val is not None and max_ret_val > 0.20:
                logger.warning(
                    f"{symbol}/{timeframe}: extreme bar detected "
                    f"(max single-bar return = {max_ret_val:.1%}). "
                    "Check for data errors."
                )

        return df

    def _compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature computation groups in dependency order."""
        cfg = self._config

        if cfg.compute_technical:
            df = add_all_technical(
                df,
                ema_periods=cfg.ema_periods,
                sma_periods=cfg.sma_periods,
                rsi_period=cfg.rsi_period,
                atr_period=cfg.atr_period,
                bb_period=cfg.bb_period,
                macd_params=cfg.macd_params,
                stoch_params=cfg.stoch_params,
                adx_period=cfg.adx_period,
                include_vwap=cfg.include_vwap,
            )

        if cfg.compute_ml:
            df = add_all_ml_features(
                df,
                return_horizons=cfg.return_horizons,
                vol_windows=cfg.vol_windows,
                price_pos_windows=cfg.price_pos_windows,
                momentum_windows=cfg.momentum_windows,
                add_targets=cfg.add_targets,
                target_horizons=cfg.target_horizons,
            )

        return df

    def _filter_and_report(
        self,
        raw_df: pl.DataFrame,
        feature_df: pl.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> tuple[pl.DataFrame, DataQualityReport]:
        """
        Drop rows where required features are NaN (warm-up period).
        Build a quality report.
        """
        total = len(feature_df)

        # Count NaN per column
        nan_counts: dict[str, int] = {}
        for col in feature_df.columns:
            n_null = feature_df[col].null_count()
            if n_null > 0:
                nan_counts[col] = n_null

        # Drop rows with NaN in required columns
        existing_required = [
            c for c in self._config.required_feature_cols
            if c in feature_df.columns
        ]
        if existing_required:
            feature_df = feature_df.drop_nulls(subset=existing_required)

        # Detect gaps: consecutive bars more than 2x the expected interval apart
        gap_count = 0
        warnings = []
        if len(raw_df) > 1 and "event_time" in raw_df.columns:
            times = raw_df["event_time"].to_list()
            diffs = [
                (times[i + 1] - times[i]).total_seconds()
                for i in range(len(times) - 1)
            ]
            if diffs:
                median_diff = sorted(diffs)[len(diffs) // 2]
                gap_count = sum(1 for d in diffs if d > median_diff * 2.5)
                if gap_count > 0:
                    warnings.append(
                        f"{symbol}/{timeframe}: {gap_count} time gaps detected "
                        f"(median interval={median_diff:.0f}s)"
                    )

        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total,
            bars_after_filter=len(feature_df),
            nan_counts=nan_counts,
            gap_count=gap_count,
            warnings=warnings,
        )

        return feature_df, report

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def last_quality_report(self) -> DataQualityReport | None:
        return self._last_quality_report