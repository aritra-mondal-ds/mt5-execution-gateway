"""
features/technical.py
---------------------
Technical indicator computation using Polars expressions.

Point-in-time correctness rules enforced here:
  - Every rolling window uses `min_periods=window` — no partial windows
    emitted. A feature is NaN until enough history exists.
  - No shift() applied here. Shifting (to prevent same-bar lookahead
    in the strategy layer) is the strategy's responsibility.
  - All inputs are sorted ascending by event_time before computation.
    The pipeline enforces this; these functions trust that contract.

Implementation philosophy:
  - Implemented from scratch with Polars expressions — no ta-lib dependency.
    This gives full transparency, easier debugging, and guaranteed
    point-in-time alignment.
  - Functions are pure: they take a DataFrame and return a DataFrame
    with new columns added. No side effects.
  - Column naming convention: indicator_period  e.g. rsi_14, atr_14, ema_21
"""

from __future__ import annotations

import polars as pl
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_df(df: pl.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def add_ema(df: pl.DataFrame, periods: list[int] = [9, 21, 50, 200]) -> pl.DataFrame:
    """
    Exponential Moving Average using Wilder's / standard span formula.
    Column names: ema_9, ema_21, ema_50, ema_200
    """
    _validate_df(df, ["close"])
    exprs = [
        pl.col("close")
          .ewm_mean(span=p, min_periods=p, ignore_nulls=True)
          .alias(f"ema_{p}")
        for p in periods
    ]
    return df.with_columns(exprs)


def add_sma(df: pl.DataFrame, periods: list[int] = [20, 50]) -> pl.DataFrame:
    """
    Simple Moving Average.
    Column names: sma_20, sma_50
    """
    _validate_df(df, ["close"])
    exprs = [
        pl.col("close")
          .rolling_mean(window_size=p, min_periods=p)
          .alias(f"sma_{p}")
        for p in periods
    ]
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Relative Strength Index — Wilder's smoothing method.

    Implementation:
      1. Compute price changes (delta)
      2. Separate gains and losses
      3. First RSI uses simple average of first `period` gains/losses
      4. Subsequent values use Wilder's smoothing:
         avg_gain = (prev_avg_gain * (period-1) + current_gain) / period
    Column name: rsi_14
    """
    _validate_df(df, ["close"])

    # Work in Pandas for the recursive Wilder smoothing — cleaner than
    # trying to express it as a Polars expression
    closes = df["close"].to_numpy().astype(np.float64)
    n = len(closes)
    rsi_values = np.full(n, np.nan)

    if n < period + 1:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"rsi_{period}"))

    delta = np.diff(closes, prepend=np.nan)   # length = n
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed: simple mean of first `period` values (index 1..period inclusive)
    first_avg_gain = np.mean(gains[1:period + 1])
    first_avg_loss = np.mean(losses[1:period + 1])

    avg_gain = first_avg_gain
    avg_loss = first_avg_loss

    if avg_loss == 0:
        rsi_values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for remaining bars
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - 100.0 / (1.0 + rs)

    # Convert numpy NaN → Polars null so drop_nulls() works correctly
    # in the pipeline. Float NaN and Polars null are NOT the same.
    return df.with_columns(
        pl.Series(name=f"rsi_{period}", values=rsi_values).fill_nan(None)
    )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Average True Range — Wilder's smoothing.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder's smoothing of TR over `period` bars.
    Column name: atr_14
    """
    _validate_df(df, ["high", "low", "close"])

    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)
    n = len(closes)

    tr = np.full(n, np.nan)
    atr_values = np.full(n, np.nan)

    # TR for first bar: just high - low (no previous close)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    if n < period:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"atr_{period}"))

    # Seed: simple mean of first `period` TR values
    atr_values[period - 1] = np.mean(tr[:period])

    # Wilder smoothing
    for i in range(period, n):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    # Convert numpy NaN → Polars null so drop_nulls() works correctly
    return df.with_columns(
        pl.Series(name=f"atr_{period}", values=atr_values).fill_nan(None)
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def add_bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.DataFrame:
    """
    Bollinger Bands: middle (SMA), upper, lower, and %B.

    Columns added:
      bb_mid_20    : SMA(close, 20)
      bb_upper_20  : mid + 2 * std
      bb_lower_20  : mid - 2 * std
      bb_pct_b_20  : (close - lower) / (upper - lower)  — 0=lower, 1=upper
      bb_width_20  : (upper - lower) / mid  — normalised bandwidth
    """
    _validate_df(df, ["close"])
    std_col = f"_bb_std_{period}"
    mid_col = f"bb_mid_{period}"
    upper_col = f"bb_upper_{period}"
    lower_col = f"bb_lower_{period}"
    pctb_col = f"bb_pct_b_{period}"
    width_col = f"bb_width_{period}"

    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=period, min_periods=period).alias(mid_col),
        pl.col("close").rolling_std(window_size=period, min_periods=period).alias(std_col),
    ])

    df = df.with_columns([
        (pl.col(mid_col) + std_dev * pl.col(std_col)).alias(upper_col),
        (pl.col(mid_col) - std_dev * pl.col(std_col)).alias(lower_col),
    ])

    df = df.with_columns([
        ((pl.col("close") - pl.col(lower_col)) /
         (pl.col(upper_col) - pl.col(lower_col))).alias(pctb_col),
        ((pl.col(upper_col) - pl.col(lower_col)) /
         pl.col(mid_col)).alias(width_col),
    ])

    return df.drop(std_col)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def add_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """
    MACD: Moving Average Convergence/Divergence.

    Columns added:
      macd_line     : EMA(fast) - EMA(slow)
      macd_signal   : EMA(macd_line, signal)
      macd_hist     : macd_line - macd_signal
    """
    _validate_df(df, ["close"])

    df = df.with_columns([
        pl.col("close").ewm_mean(span=fast, min_periods=fast, ignore_nulls=True).alias("_ema_fast"),
        pl.col("close").ewm_mean(span=slow, min_periods=slow, ignore_nulls=True).alias("_ema_slow"),
    ])

    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
    )

    df = df.with_columns(
        pl.col("macd_line")
          .ewm_mean(span=signal, min_periods=signal, ignore_nulls=True)
          .alias("macd_signal")
    )

    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )

    return df.drop(["_ema_fast", "_ema_slow"])


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def add_stochastic(
    df: pl.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pl.DataFrame:
    """
    Stochastic Oscillator %K and %D.

    %K = (close - lowest_low_k) / (highest_high_k - lowest_low_k) * 100
    %D = SMA(%K, d_period)

    Columns: stoch_k_14, stoch_d_14
    """
    _validate_df(df, ["high", "low", "close"])

    df = df.with_columns([
        pl.col("high").rolling_max(window_size=k_period, min_periods=k_period).alias("_hh"),
        pl.col("low").rolling_min(window_size=k_period, min_periods=k_period).alias("_ll"),
    ])

    df = df.with_columns(
        ((pl.col("close") - pl.col("_ll")) /
         (pl.col("_hh") - pl.col("_ll")) * 100.0).alias(f"stoch_k_{k_period}")
    )

    df = df.with_columns(
        pl.col(f"stoch_k_{k_period}")
          .rolling_mean(window_size=d_period, min_periods=d_period)
          .alias(f"stoch_d_{k_period}")
    )

    return df.drop(["_hh", "_ll"])


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def add_adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Average Directional Index — Wilder's method.

    Measures trend strength (not direction). >25 = strong trend.

    Columns added:
      adx_14   : Average Directional Index
      pdi_14   : +DI (bullish directional indicator)
      ndi_14   : -DI (bearish directional indicator)
    """
    _validate_df(df, ["high", "low", "close"])

    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)
    n = len(closes)

    tr_arr = np.full(n, np.nan)
    pdm_arr = np.full(n, np.nan)   # +DM
    ndm_arr = np.full(n, np.nan)   # -DM

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr_arr[i] = max(hl, hpc, lpc)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        pdm_arr[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        ndm_arr[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    atr_s = np.full(n, np.nan)
    pdm_s = np.full(n, np.nan)
    ndm_s = np.full(n, np.nan)
    pdi = np.full(n, np.nan)
    ndi = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    if n <= period:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(f"adx_{period}"),
            pl.lit(None).cast(pl.Float64).alias(f"pdi_{period}"),
            pl.lit(None).cast(pl.Float64).alias(f"ndi_{period}"),
        ])

    # Seed smoothed values with simple sum of first `period` values
    atr_s[period] = np.nansum(tr_arr[1:period + 1])
    pdm_s[period] = np.nansum(pdm_arr[1:period + 1])
    ndm_s[period] = np.nansum(ndm_arr[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr_arr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + pdm_arr[i]
        ndm_s[i] = ndm_s[i - 1] - ndm_s[i - 1] / period + ndm_arr[i]

    for i in range(period, n):
        if atr_s[i] != 0:
            pdi[i] = 100.0 * pdm_s[i] / atr_s[i]
            ndi[i] = 100.0 * ndm_s[i] / atr_s[i]
            denom = pdi[i] + ndi[i]
            if denom != 0:
                dx[i] = 100.0 * abs(pdi[i] - ndi[i]) / denom

    # ADX = Wilder smoothing of DX
    if np.any(~np.isnan(dx[period:])):
        first_valid = period
        adx[first_valid + period - 1] = np.nanmean(dx[first_valid:first_valid + period])
        for i in range(first_valid + period, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    # Convert numpy NaN → Polars null so drop_nulls() works correctly
    return df.with_columns([
        pl.Series(name=f"adx_{period}", values=adx).fill_nan(None),
        pl.Series(name=f"pdi_{period}", values=pdi).fill_nan(None),
        pl.Series(name=f"ndi_{period}", values=ndi).fill_nan(None),
    ])


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def add_vwap(df: pl.DataFrame, session_col: str | None = None) -> pl.DataFrame:
    """
    Volume-Weighted Average Price.

    For forex/crypto (no true volume), we use tick_volume as a proxy.
    VWAP resets each trading session (day by default).

    Uses typical price = (high + low + close) / 3

    Columns added:
      vwap              : running VWAP
      vwap_distance     : (close - vwap) / vwap  — normalised distance
    """
    _validate_df(df, ["high", "low", "close", "volume", "event_time"])

    df = df.with_columns([
        # Typical price
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp"),
        # Session identifier: date portion of event_time
        pl.col("event_time").dt.date().alias("_session"),
    ])

    # Cumulative sum within each session using group_by + cumulative
    df = df.with_columns([
        (pl.col("_tp") * pl.col("volume")).alias("_tp_vol"),
    ])

    # Polars: cumulative sum partitioned by session
    df = df.with_columns([
        pl.col("_tp_vol")
          .cum_sum()
          .over("_session")
          .alias("_cum_tp_vol"),
        pl.col("volume")
          .cum_sum()
          .over("_session")
          .alias("_cum_vol"),
    ])

    df = df.with_columns(
        (pl.col("_cum_tp_vol") / pl.col("_cum_vol")).alias("vwap")
    )

    df = df.with_columns(
        ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_distance")
    )

    return df.drop(["_tp", "_session", "_tp_vol", "_cum_tp_vol", "_cum_vol"])


# ---------------------------------------------------------------------------
# Composite: add all technical features at once
# ---------------------------------------------------------------------------

def add_all_technical(
    df: pl.DataFrame,
    ema_periods: list[int] = [9, 21, 50, 200],
    sma_periods: list[int] = [20, 50],
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    stoch_params: tuple[int, int] = (14, 3),
    adx_period: int = 14,
    include_vwap: bool = True,
) -> pl.DataFrame:
    """
    Apply the full technical indicator suite in a single call.
    Returns the input DataFrame with all indicator columns appended.

    Minimum bars required for all indicators: max(200, 26+9, 14+14) = 200
    (driven by ema_200)
    """
    n_bars = len(df)
    logger.debug(f"Computing technical features on {n_bars} bars")

    df = add_ema(df, ema_periods)
    df = add_sma(df, sma_periods)
    df = add_rsi(df, rsi_period)
    df = add_atr(df, atr_period)
    df = add_bollinger_bands(df, bb_period)
    df = add_macd(df, *macd_params)
    df = add_stochastic(df, *stoch_params)
    df = add_adx(df, adx_period)

    if include_vwap and "volume" in df.columns:
        df = add_vwap(df)

    logger.debug(f"Technical features complete. Columns: {df.columns}")
    return df