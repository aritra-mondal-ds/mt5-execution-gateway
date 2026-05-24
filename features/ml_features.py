"""
features/ml_features.py
------------------------
ML-oriented features derived from raw OHLCV data.

These features are designed to be inputs to ML models (LightGBM, LSTM).
They complement the technical indicators in technical.py.

Categories:
  1. Return features       — log returns over multiple horizons
  2. Volatility features   — realised vol, vol regime, vol ratio
  3. Bar pattern features  — body/wick ratios, inside bars, engulfing
  4. Price position        — relative to recent range, EMA distances
  5. Momentum features     — autocorrelation, trend consistency

All features are backward-looking only.
No future information is used at any row.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger


# ---------------------------------------------------------------------------
# 1. Return features
# ---------------------------------------------------------------------------

def add_returns(
    df: pl.DataFrame,
    horizons: list[int] = [1, 2, 5, 10, 20],
) -> pl.DataFrame:
    """
    Log returns over multiple look-back horizons.

    log_return_n = log(close_t / close_{t-n})

    Using log returns because:
      - Additive over time (compounding is just addition)
      - More normally distributed than simple returns
      - Better behaved for ML models

    Columns: log_ret_1, log_ret_5, log_ret_10, log_ret_20
    """
    exprs = [
        (pl.col("close") / pl.col("close").shift(n)).log(base=np.e).alias(f"log_ret_{n}")
        for n in horizons
    ]
    return df.with_columns(exprs)


def add_future_returns(
    df: pl.DataFrame,
    horizons: list[int] = [1, 5, 10],
) -> pl.DataFrame:
    """
    Forward returns — TARGET VARIABLES for supervised ML training.

    IMPORTANT: These columns must NEVER be used as features.
    They are labels only. The pipeline flags them with 'target_' prefix.

    In backtesting and live trading, these are computed on historical
    data for training only, never at inference time.

    Columns: target_ret_1, target_ret_5, target_ret_10
    """
    exprs = [
        (pl.col("close").shift(-n) / pl.col("close")).log(base=np.e).alias(f"target_ret_{n}")
        for n in horizons
    ]
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# 2. Volatility features
# ---------------------------------------------------------------------------

def add_realized_volatility(
    df: pl.DataFrame,
    windows: list[int] = [5, 10, 20, 60],
    annualization_factor: float = 252.0,
) -> pl.DataFrame:
    """
    Realized volatility: rolling std of log returns, annualized.

    rv_n = std(log_ret_1, window=n) * sqrt(annualization_factor)

    Requires log_ret_1 to exist (call add_returns first).
    Columns: rv_5, rv_10, rv_20, rv_60
    """
    if "log_ret_1" not in df.columns:
        df = add_returns(df, horizons=[1])

    ann_sqrt = np.sqrt(annualization_factor)
    exprs = [
        (pl.col("log_ret_1").rolling_std(window_size=w, min_periods=w) * ann_sqrt)
        .alias(f"rv_{w}")
        for w in windows
    ]
    return df.with_columns(exprs)


def add_volatility_regime(
    df: pl.DataFrame,
    short_window: int = 10,
    long_window: int = 60,
) -> pl.DataFrame:
    """
    Volatility regime features.

    vol_ratio    : rv_short / rv_long  — >1 means vol expanding
    vol_zscore   : (rv_short - mean(rv_short, long)) / std(rv_short, long)
                   — how unusual current vol is vs recent history
    vol_regime   : 0=low, 1=normal, 2=high  (based on zscore thresholds)

    Requires rv_{short_window} and rv_{long_window} columns.
    """
    short_col = f"rv_{short_window}"
    long_col = f"rv_{long_window}"

    if short_col not in df.columns or long_col not in df.columns:
        df = add_realized_volatility(df, windows=[short_window, long_window])

    df = df.with_columns([
        (pl.col(short_col) / pl.col(long_col)).alias("vol_ratio"),
        (
            (pl.col(short_col) - pl.col(short_col).rolling_mean(long_window, min_periods=long_window)) /
            pl.col(short_col).rolling_std(long_window, min_periods=long_window)
        ).alias("vol_zscore"),
    ])

    # Discretize into regime: low(<-1), normal(-1 to 1), high(>1)
    df = df.with_columns(
        pl.when(pl.col("vol_zscore") < -1.0).then(0)
          .when(pl.col("vol_zscore") > 1.0).then(2)
          .otherwise(1)
          .alias("vol_regime")
    )

    return df


def add_atr_ratio(
    df: pl.DataFrame,
    short: int = 7,
    long: int = 28,
) -> pl.DataFrame:
    """
    ATR ratio: short ATR / long ATR.
    Normalises current bar's range against recent average range.
    Useful as a regime indicator that doesn't require returns.

    Also adds: bar_range_ratio = (high-low) / atr_long
    """
    from features.technical import add_atr

    if f"atr_{short}" not in df.columns:
        df = add_atr(df, short)
    if f"atr_{long}" not in df.columns:
        df = add_atr(df, long)

    df = df.with_columns([
        (pl.col(f"atr_{short}") / pl.col(f"atr_{long}")).alias("atr_ratio"),
        ((pl.col("high") - pl.col("low")) / pl.col(f"atr_{long}")).alias("bar_range_ratio"),
    ])

    return df


# ---------------------------------------------------------------------------
# 3. Bar pattern features
# ---------------------------------------------------------------------------

def add_bar_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Price action features derived from a single bar's OHLC.

    body_ratio      : |close - open| / (high - low)   — how much of range is body
                      0 = doji, 1 = no wicks
    upper_wick_ratio: (high - max(open,close)) / (high - low)
    lower_wick_ratio: (min(open,close) - low) / (high - low)
    bar_direction   : +1 if bullish, -1 if bearish, 0 if doji
    rel_close       : (close - low) / (high - low)    — where close sits in range
                      0 = closed at low, 1 = closed at high
    """
    _range = pl.col("high") - pl.col("low")
    _body = (pl.col("close") - pl.col("open")).abs()
    _upper_wick = pl.col("high") - pl.max_horizontal("open", "close")
    _lower_wick = pl.min_horizontal("open", "close") - pl.col("low")

    # Avoid division by zero on doji bars with zero range
    _safe_range = pl.when(_range == 0).then(1e-10).otherwise(_range)

    return df.with_columns([
        (_body / _safe_range).alias("body_ratio"),
        (_upper_wick / _safe_range).alias("upper_wick_ratio"),
        (_lower_wick / _safe_range).alias("lower_wick_ratio"),
        (
            pl.when(pl.col("close") > pl.col("open")).then(1)
              .when(pl.col("close") < pl.col("open")).then(-1)
              .otherwise(0)
        ).alias("bar_direction"),
        ((pl.col("close") - pl.col("low")) / _safe_range).alias("rel_close"),
    ])


def add_candlestick_patterns(df: pl.DataFrame, atr_period: int = 14) -> pl.DataFrame:
    """
    Binary candlestick pattern flags. Each is 1 if pattern present, 0 if not.

    Patterns:
      - doji          : body < 10% of range
      - hammer        : lower wick > 2x body, small upper wick, body in upper 1/3
      - shooting_star : upper wick > 2x body, small lower wick, body in lower 1/3
      - inside_bar    : high < prev_high AND low > prev_low
      - engulfing_bull: bearish prev bar fully engulfed by bullish current bar
      - engulfing_bear: bullish prev bar fully engulfed by bearish current bar
    """
    if "body_ratio" not in df.columns:
        df = add_bar_patterns(df)

    _range = pl.col("high") - pl.col("low")
    _body = (pl.col("close") - pl.col("open")).abs()
    _safe_range = pl.when(_range == 0).then(1e-10).otherwise(_range)
    _upper_wick = pl.col("high") - pl.max_horizontal("open", "close")
    _lower_wick = pl.min_horizontal("open", "close") - pl.col("low")

    # Note: patterns using shift(1) produce null on the first row.
    # We fill_null(0) — no pattern at the first bar is correct (no prior bar).
    return df.with_columns([
        # Doji: body < 10% of range (no shift → no null)
        (pl.col("body_ratio") < 0.10).cast(pl.Int8).alias("pat_doji"),

        # Hammer: lower wick >= 2x body, upper wick < body, bullish lean
        (
            (_lower_wick >= 2.0 * _body) &
            (_upper_wick <= _body) &
            (pl.col("rel_close") > 0.5)
        ).cast(pl.Int8).alias("pat_hammer"),

        # Shooting star: upper wick >= 2x body, lower wick < body, bearish lean
        (
            (_upper_wick >= 2.0 * _body) &
            (_lower_wick <= _body) &
            (pl.col("rel_close") < 0.5)
        ).cast(pl.Int8).alias("pat_shooting_star"),

        # Inside bar: uses shift(1) → fill null first row with 0
        (
            (pl.col("high") < pl.col("high").shift(1)) &
            (pl.col("low") > pl.col("low").shift(1))
        ).cast(pl.Int8).fill_null(0).alias("pat_inside_bar"),

        # Bullish engulfing: uses shift(1) → fill null first row with 0
        (
            (pl.col("close").shift(1) < pl.col("open").shift(1)) &  # prev bearish
            (pl.col("close") > pl.col("open")) &                     # curr bullish
            (pl.col("open") <= pl.col("close").shift(1)) &           # opens below prev close
            (pl.col("close") >= pl.col("open").shift(1))             # closes above prev open
        ).cast(pl.Int8).fill_null(0).alias("pat_engulfing_bull"),

        # Bearish engulfing: uses shift(1) → fill null first row with 0
        (
            (pl.col("close").shift(1) > pl.col("open").shift(1)) &  # prev bullish
            (pl.col("close") < pl.col("open")) &                     # curr bearish
            (pl.col("open") >= pl.col("close").shift(1)) &           # opens above prev close
            (pl.col("close") <= pl.col("open").shift(1))             # closes below prev open
        ).cast(pl.Int8).fill_null(0).alias("pat_engulfing_bear"),
    ])


# ---------------------------------------------------------------------------
# 4. Price position features
# ---------------------------------------------------------------------------

def add_price_position(
    df: pl.DataFrame,
    windows: list[int] = [10, 20, 50],
) -> pl.DataFrame:
    """
    Where is the current close relative to recent price history?

    For each window n:
      price_pos_n  : (close - min_low_n) / (max_high_n - min_low_n)
                     0 = at n-bar low, 1 = at n-bar high
      dist_from_high_n : (max_high_n - close) / max_high_n
      dist_from_low_n  : (close - min_low_n) / close
    """
    exprs = []
    for w in windows:
        max_high = pl.col("high").rolling_max(window_size=w, min_periods=w)
        min_low = pl.col("low").rolling_min(window_size=w, min_periods=w)
        price_range = max_high - min_low
        safe_range = pl.when(price_range == 0).then(1e-10).otherwise(price_range)

        exprs.extend([
            ((pl.col("close") - min_low) / safe_range).alias(f"price_pos_{w}"),
            ((max_high - pl.col("close")) / max_high).alias(f"dist_high_{w}"),
            ((pl.col("close") - min_low) / pl.col("close")).alias(f"dist_low_{w}"),
        ])

    return df.with_columns(exprs)


def add_ema_distances(
    df: pl.DataFrame,
    periods: list[int] = [9, 21, 50, 200],
) -> pl.DataFrame:
    """
    Distance of close from each EMA, normalised by close price.

    ema_dist_n = (close - ema_n) / close

    Positive = close above EMA (bullish), negative = below (bearish).
    Requires EMA columns to exist (call add_ema first, or add_all_technical).
    """
    exprs = []
    for p in periods:
        col = f"ema_{p}"
        if col in df.columns:
            exprs.append(
                ((pl.col("close") - pl.col(col)) / pl.col("close")).alias(f"ema_dist_{p}")
            )
    if exprs:
        df = df.with_columns(exprs)
    return df


# ---------------------------------------------------------------------------
# 5. Momentum / autocorrelation features
# ---------------------------------------------------------------------------

def add_momentum_features(
    df: pl.DataFrame,
    windows: list[int] = [5, 10, 20],
) -> pl.DataFrame:
    """
    Momentum and trend consistency features.

    For each window n:
      trend_consistency_n : fraction of bars in window that moved in same
                            direction as current bar's log return
                            (0=all against, 0.5=random, 1=all same)
      roc_n               : Rate of Change = (close / close_n_bars_ago - 1) * 100
    """
    if "log_ret_1" not in df.columns:
        df = add_returns(df, horizons=[1])

    exprs = []
    for w in windows:
        # Rate of Change
        exprs.append(
            ((pl.col("close") / pl.col("close").shift(w) - 1.0) * 100.0)
            .alias(f"roc_{w}")
        )

    df = df.with_columns(exprs)

    # Trend consistency: sign of log_ret_1 agrees with rolling direction
    # = rolling_mean(sign(ret)) shifted to avoid lookahead
    df = df.with_columns([
        pl.col("log_ret_1").sign().cast(pl.Float64).alias("_ret_sign")
    ])

    consistency_exprs = [
        ((pl.col("_ret_sign").rolling_mean(window_size=w, min_periods=w) + 1.0) / 2.0)
        .alias(f"trend_consistency_{w}")
        for w in windows
    ]
    df = df.with_columns(consistency_exprs)
    return df.drop("_ret_sign")


# ---------------------------------------------------------------------------
# Composite: add all ML features
# ---------------------------------------------------------------------------

def add_all_ml_features(
    df: pl.DataFrame,
    return_horizons: list[int] = [1, 2, 5, 10, 20],
    vol_windows: list[int] = [5, 10, 20, 60],
    price_pos_windows: list[int] = [10, 20, 50],
    momentum_windows: list[int] = [5, 10, 20],
    add_targets: bool = False,
    target_horizons: list[int] = [1, 5, 10],
) -> pl.DataFrame:
    """
    Apply the full ML feature suite.

    Set add_targets=True only during training data preparation.
    NEVER set add_targets=True in live inference or backtesting.
    """
    n_bars = len(df)
    logger.debug(f"Computing ML features on {n_bars} bars")

    df = add_returns(df, return_horizons)
    df = add_realized_volatility(df, vol_windows)
    df = add_volatility_regime(df)
    df = add_atr_ratio(df)
    df = add_bar_patterns(df)
    df = add_candlestick_patterns(df)
    df = add_price_position(df, price_pos_windows)
    df = add_ema_distances(df)
    df = add_momentum_features(df, momentum_windows)

    if add_targets:
        logger.warning(
            "add_targets=True: target columns added. "
            "These are LABELS ONLY — never use as features."
        )
        df = add_future_returns(df, target_horizons)

    logger.debug(f"ML features complete. Total columns: {len(df.columns)}")
    return df