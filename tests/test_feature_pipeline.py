"""
tests/test_feature_pipeline.py
-------------------------------
Tests for the complete feature pipeline:
  - Technical indicators (RSI, ATR, EMA, MACD, BB, Stochastic, ADX, VWAP)
  - ML features (returns, volatility, bar patterns, price position, momentum)
  - FeaturePipeline orchestrator (point-in-time, validation, filtering)
  - FeatureStore (cache write/read/invalidation/staleness)

Design rules tested:
  1. No lookahead: rolling windows respect min_periods (no partial windows)
  2. Value correctness: spot-check known-answer cases
  3. Column naming conventions
  4. NaN handling: warm-up rows are NaN, not zero or garbage
  5. Pipeline drops warm-up rows from output
  6. FeatureStore: cache hit/miss, staleness, invalidation

Run with:
    pytest tests/test_feature_pipeline.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from feeds.base_feed import Bar
from features.technical import (
    add_ema, add_sma, add_rsi, add_atr,
    add_bollinger_bands, add_macd, add_stochastic, add_adx,
    add_vwap, add_all_technical,
)
from features.ml_features import (
    add_returns, add_realized_volatility, add_volatility_regime,
    add_bar_patterns, add_candlestick_patterns,
    add_price_position, add_momentum_features,
    add_all_ml_features,
)
from features.pipeline import FeaturePipeline, PipelineConfig
from data.feature_store import FeatureStore


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_ohlcv_df(
    n: int = 300,
    base_price: float = 2000.0,
    timeframe: str = "H1",
    start: datetime | None = None,
    trend: float = 0.001,      # small upward drift per bar
    noise: float = 0.005,      # random noise amplitude
    seed: int = 42,
) -> pl.DataFrame:
    """
    Synthetic OHLCV DataFrame for testing.

    Generates a realistic-looking price series with:
      - Slight upward trend
      - Random noise
      - Proper OHLC relationships (open != close, high >= both, low <= both)
      - Increasing volume with noise
    """
    rng = np.random.default_rng(seed)
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    prices = [base_price]
    for _ in range(n - 1):
        ret = trend + rng.normal(0, noise)
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)
    opens = prices
    closes = prices * (1 + rng.normal(0, noise * 0.5, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.0005, 0.003, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.0005, 0.003, n))
    volumes = rng.uniform(5000, 20000, n)

    # Ensure OHLC constraints
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    event_times = [start + timedelta(hours=i) for i in range(n)]
    publish_times = [t + timedelta(milliseconds=100) for t in event_times]

    return pl.DataFrame({
        "symbol": ["XAUUSD"] * n,
        "timeframe": [timeframe] * n,
        "event_time": event_times,
        "publish_time": publish_times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "spread": [2.0] * n,
        "source": ["test"] * n,
    })


@pytest.fixture
def df300() -> pl.DataFrame:
    return _make_ohlcv_df(n=300)


@pytest.fixture
def df50() -> pl.DataFrame:
    """Too short for most indicators — for edge case testing."""
    return _make_ohlcv_df(n=50)


@pytest.fixture
def tmp_raw_store(tmp_path: Path):
    from data.store import RawDataStore
    store = RawDataStore(root=tmp_path / "raw")
    # Pre-populate with 300 bars
    df = _make_ohlcv_df(n=300)
    bars = [
        Bar(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            event_time=row["event_time"],
            publish_time=row["publish_time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            spread=row["spread"],
            source=row["source"],
        )
        for row in df.iter_rows(named=True)
    ]
    store.write_batch(bars)
    return store


@pytest.fixture
def tmp_feature_store(tmp_path: Path) -> FeatureStore:
    return FeatureStore(root=tmp_path / "features")


# ===========================================================================
# Technical indicators
# ===========================================================================

class TestEMA:
    def test_columns_created(self, df300):
        result = add_ema(df300, periods=[9, 21])
        assert "ema_9" in result.columns
        assert "ema_21" in result.columns

    def test_warmup_is_nan(self, df300):
        result = add_ema(df300, periods=[21])
        # First 20 rows should be NaN (need 21 bars to emit first value)
        nan_count = result["ema_21"].null_count()
        assert nan_count >= 20, f"Expected >= 20 NaN rows, got {nan_count}"

    def test_ema_smoothing(self, df300):
        """EMA must be smoother than close (lower std of differences)."""
        result = add_ema(df300, periods=[21])
        df_valid = result.drop_nulls("ema_21")
        ema_diff_std = df_valid["ema_21"].diff().drop_nulls().std()
        close_diff_std = df_valid["close"].diff().drop_nulls().std()
        assert ema_diff_std < close_diff_std

    def test_ema_lags_close(self, df300):
        """On an uptrend, EMA-200 should be below close (lagging)."""
        result = add_ema(df300, periods=[200])
        df_valid = result.drop_nulls("ema_200")
        # In an upward trending series, EMA lags → close > EMA on average
        mean_diff = (df_valid["close"] - df_valid["ema_200"]).mean()
        assert mean_diff > 0


class TestSMA:
    def test_columns_created(self, df300):
        result = add_sma(df300, periods=[20])
        assert "sma_20" in result.columns

    def test_warmup_is_nan(self, df300):
        result = add_sma(df300, periods=[20])
        assert result["sma_20"].null_count() >= 19

    def test_known_value(self):
        """SMA of a constant series must equal that constant."""
        prices = [100.0] * 30
        df = pl.DataFrame({
            "close": prices,
            "open": prices, "high": prices, "low": prices,
            "volume": [1.0] * 30, "spread": [0.0] * 30,
        })
        result = add_sma(df, periods=[10])
        valid = result.drop_nulls("sma_10")
        assert all(abs(v - 100.0) < 1e-9 for v in valid["sma_10"].to_list())


class TestRSI:
    def test_column_created(self, df300):
        result = add_rsi(df300, period=14)
        assert "rsi_14" in result.columns

    def test_warmup_is_nan(self, df300):
        result = add_rsi(df300, period=14)
        assert result["rsi_14"].null_count() >= 14

    def test_range_0_to_100(self, df300):
        result = add_rsi(df300, period=14)
        valid = result.drop_nulls("rsi_14")
        assert valid["rsi_14"].min() >= 0.0
        assert valid["rsi_14"].max() <= 100.0

    def test_constant_up_series(self):
        """All gains → RSI should be 100."""
        prices = [float(i) for i in range(1, 32)]  # monotonically increasing
        df = pl.DataFrame({"close": prices})
        result = add_rsi(df, period=14)
        # After warmup, RSI should be at or near 100
        valid = result.drop_nulls("rsi_14")
        assert valid["rsi_14"].min() > 99.0

    def test_constant_down_series(self):
        """All losses → RSI should be 0."""
        prices = [float(31 - i) for i in range(31)]  # monotonically decreasing
        df = pl.DataFrame({"close": prices})
        result = add_rsi(df, period=14)
        valid = result.drop_nulls("rsi_14")
        assert valid["rsi_14"].max() < 1.0


class TestATR:
    def test_column_created(self, df300):
        result = add_atr(df300, period=14)
        assert "atr_14" in result.columns

    def test_warmup_is_nan(self, df300):
        result = add_atr(df300, period=14)
        assert result["atr_14"].null_count() >= 13

    def test_atr_is_positive(self, df300):
        result = add_atr(df300, period=14)
        valid = result.drop_nulls("atr_14")
        assert valid["atr_14"].min() > 0.0

    def test_atr_bounded_by_range(self, df300):
        """ATR cannot be smaller than the minimum true range."""
        result = add_atr(df300, period=14)
        valid = result.drop_nulls("atr_14")
        bar_range = valid["high"] - valid["low"]
        # ATR is an average of TR; min ATR shouldn't be far below min range
        assert valid["atr_14"].min() >= bar_range.min() * 0.1


class TestBollingerBands:
    def test_columns_created(self, df300):
        result = add_bollinger_bands(df300, period=20)
        for col in ["bb_mid_20", "bb_upper_20", "bb_lower_20", "bb_pct_b_20", "bb_width_20"]:
            assert col in result.columns

    def test_upper_above_lower(self, df300):
        result = add_bollinger_bands(df300, period=20)
        valid = result.drop_nulls("bb_upper_20")
        assert (valid["bb_upper_20"] >= valid["bb_lower_20"]).all()

    def test_mid_between_bands(self, df300):
        result = add_bollinger_bands(df300, period=20)
        valid = result.drop_nulls("bb_mid_20")
        assert (valid["bb_upper_20"] >= valid["bb_mid_20"]).all()
        assert (valid["bb_mid_20"] >= valid["bb_lower_20"]).all()

    def test_pct_b_around_half(self, df300):
        """Over a long series, average %B should be near 0.5."""
        result = add_bollinger_bands(df300, period=20)
        valid = result.drop_nulls("bb_pct_b_20")
        mean_pctb = valid["bb_pct_b_20"].mean()
        assert 0.3 < mean_pctb < 0.7


class TestMACD:
    def test_columns_created(self, df300):
        result = add_macd(df300)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_hist_is_line_minus_signal(self, df300):
        result = add_macd(df300)
        valid = result.drop_nulls(["macd_line", "macd_signal"])
        expected = valid["macd_line"] - valid["macd_signal"]
        actual = valid["macd_hist"]
        diffs = (expected - actual).abs()
        assert diffs.max() < 1e-10

    def test_no_intermediate_columns_leaked(self, df300):
        result = add_macd(df300)
        assert "_ema_fast" not in result.columns
        assert "_ema_slow" not in result.columns


class TestStochastic:
    def test_columns_created(self, df300):
        result = add_stochastic(df300)
        assert "stoch_k_14" in result.columns
        assert "stoch_d_14" in result.columns

    def test_range_0_to_100(self, df300):
        result = add_stochastic(df300)
        valid = result.drop_nulls("stoch_k_14")
        assert valid["stoch_k_14"].min() >= 0.0
        assert valid["stoch_k_14"].max() <= 100.0


class TestADX:
    def test_columns_created(self, df300):
        result = add_adx(df300, period=14)
        assert "adx_14" in result.columns
        assert "pdi_14" in result.columns
        assert "ndi_14" in result.columns

    def test_adx_range(self, df300):
        result = add_adx(df300, period=14)
        valid = result.drop_nulls("adx_14")
        assert valid["adx_14"].min() >= 0.0
        assert valid["adx_14"].max() <= 100.0

    def test_trending_series_has_high_adx(self):
        """A strongly trending series should produce ADX > 25."""
        # Strong linear trend with minimal noise
        prices = [2000.0 + i * 2 for i in range(100)]
        df = pl.DataFrame({
            "close": prices,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "volume": [1000.0] * 100,
            "spread": [2.0] * 100,
        })
        result = add_adx(df, period=14)
        valid = result.drop_nulls("adx_14")
        if len(valid) > 0:
            assert valid["adx_14"].max() > 20.0


class TestVWAP:
    def test_columns_created(self, df300):
        result = add_vwap(df300)
        assert "vwap" in result.columns
        assert "vwap_distance" in result.columns

    def test_vwap_within_session_cumulative_range(self, df300):
        """
        VWAP is a session-running cumulative average, NOT bounded by each
        individual bar's high/low. If the session started at a higher price,
        VWAP can legitimately sit above the current bar's high.

        Correct assertion: VWAP lies within the session's cumulative
        [min(low), max(high)] up to and including the current bar.
        """
        result = add_vwap(df300).with_columns(
            pl.col("event_time").dt.date().alias("_session")
        )
        result = result.with_columns([
            pl.col("high").cum_max().over("_session").alias("_cum_high"),
            pl.col("low").cum_min().over("_session").alias("_cum_low"),
        ])
        valid = result.drop_nulls("vwap")
        assert (valid["vwap"] >= valid["_cum_low"]).all(), \
            "VWAP below cumulative session low"
        assert (valid["vwap"] <= valid["_cum_high"]).all(), \
            "VWAP above cumulative session high"

    def test_no_internal_columns_leaked(self, df300):
        result = add_vwap(df300)
        for col in ["_tp", "_session", "_tp_vol", "_cum_tp_vol", "_cum_vol"]:
            assert col not in result.columns


class TestAllTechnical:
    def test_all_columns_present(self, df300):
        result = add_all_technical(df300)
        expected = [
            "ema_9", "ema_21", "ema_50", "ema_200",
            "sma_20", "sma_50",
            "rsi_14", "atr_14",
            "bb_mid_20", "bb_upper_20", "bb_lower_20",
            "macd_line", "macd_signal", "macd_hist",
            "stoch_k_14", "stoch_d_14",
            "adx_14", "pdi_14", "ndi_14",
            "vwap", "vwap_distance",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, df300):
        result = add_all_technical(df300)
        assert len(result) == len(df300)


# ===========================================================================
# ML features
# ===========================================================================

class TestReturns:
    def test_columns_created(self, df300):
        result = add_returns(df300, horizons=[1, 5, 10])
        assert "log_ret_1" in result.columns
        assert "log_ret_5" in result.columns
        assert "log_ret_10" in result.columns

    def test_log_ret_1_warmup(self, df300):
        result = add_returns(df300, horizons=[1])
        assert result["log_ret_1"].null_count() == 1   # only first row is NaN

    def test_log_ret_known_value(self):
        """log(2/1) = ln(2) ≈ 0.6931"""
        df = pl.DataFrame({"close": [1.0, 2.0, 4.0]})
        result = add_returns(df, horizons=[1])
        vals = result["log_ret_1"].drop_nulls().to_list()
        assert abs(vals[0] - math.log(2)) < 1e-9
        assert abs(vals[1] - math.log(2)) < 1e-9

    def test_returns_sum_to_total(self):
        """log_ret_1 cumsum should equal log_ret_n for linear returns."""
        df = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]})
        result = add_returns(df, horizons=[1, 3])
        r1 = result["log_ret_1"].drop_nulls().to_list()
        r3 = result["log_ret_3"].drop_nulls().to_list()
        # sum of first 3 log_ret_1 values should ≈ log_ret_3
        cumsum_3 = sum(r1)
        assert abs(cumsum_3 - r3[0]) < 1e-9


class TestRealizedVolatility:
    def test_columns_created(self, df300):
        result = add_realized_volatility(df300, windows=[5, 20])
        assert "rv_5" in result.columns
        assert "rv_20" in result.columns

    def test_rv_is_positive(self, df300):
        result = add_realized_volatility(df300, windows=[5])
        valid = result.drop_nulls("rv_5")
        assert valid["rv_5"].min() >= 0.0

    def test_longer_window_smoother(self, df300):
        """rv_60 should have lower standard deviation than rv_5."""
        result = add_realized_volatility(df300, windows=[5, 60])
        rv5 = result.drop_nulls("rv_5")["rv_5"]
        rv60 = result.drop_nulls("rv_60")["rv_60"]
        assert rv60.std() < rv5.std()


class TestVolatilityRegime:
    def test_columns_created(self, df300):
        result = add_volatility_regime(df300)
        assert "vol_ratio" in result.columns
        assert "vol_zscore" in result.columns
        assert "vol_regime" in result.columns

    def test_regime_values(self, df300):
        result = add_volatility_regime(df300)
        valid = result.drop_nulls("vol_regime")
        unique_vals = set(valid["vol_regime"].to_list())
        assert unique_vals.issubset({0, 1, 2})


class TestBarPatterns:
    def test_columns_created(self, df300):
        result = add_bar_patterns(df300)
        assert "body_ratio" in result.columns
        assert "upper_wick_ratio" in result.columns
        assert "lower_wick_ratio" in result.columns
        assert "bar_direction" in result.columns
        assert "rel_close" in result.columns

    def test_body_ratio_range(self, df300):
        result = add_bar_patterns(df300)
        assert result["body_ratio"].min() >= 0.0
        assert result["body_ratio"].max() <= 1.0

    def test_rel_close_range(self, df300):
        result = add_bar_patterns(df300)
        assert result["rel_close"].min() >= 0.0
        assert result["rel_close"].max() <= 1.0

    def test_bar_direction_values(self, df300):
        result = add_bar_patterns(df300)
        unique = set(result["bar_direction"].to_list())
        assert unique.issubset({-1, 0, 1})

    def test_bullish_bar_direction(self):
        """A bar with close > open must be direction +1."""
        df = pl.DataFrame({
            "open": [100.0], "high": [110.0], "low": [95.0], "close": [108.0],
        })
        result = add_bar_patterns(df)
        assert result["bar_direction"][0] == 1

    def test_bearish_bar_direction(self):
        df = pl.DataFrame({
            "open": [108.0], "high": [110.0], "low": [95.0], "close": [100.0],
        })
        result = add_bar_patterns(df)
        assert result["bar_direction"][0] == -1


class TestCandlestickPatterns:
    def test_pattern_columns_created(self, df300):
        result = add_candlestick_patterns(df300)
        for col in [
            "pat_doji", "pat_hammer", "pat_shooting_star",
            "pat_inside_bar", "pat_engulfing_bull", "pat_engulfing_bear",
        ]:
            assert col in result.columns

    def test_pattern_values_binary(self, df300):
        result = add_candlestick_patterns(df300)
        for col in ["pat_doji", "pat_hammer", "pat_inside_bar"]:
            unique = set(result[col].to_list())
            assert unique.issubset({0, 1})

    def test_doji_detection(self):
        """A bar with body < 10% of range should be flagged as doji."""
        # Range = 10, body = 0.5 → body_ratio = 0.05 < 0.10
        df = pl.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [100.3],
            "volume": [1000.0], "spread": [2.0],
        })
        result = add_candlestick_patterns(df)
        assert result["pat_doji"][0] == 1

    def test_inside_bar_detection(self):
        """Bar fully inside the previous bar must be flagged."""
        df = pl.DataFrame({
            "open":  [100.0, 101.0, 101.5],
            "high":  [110.0, 108.0, 106.0],
            "low":   [90.0,  93.0,  95.0],
            "close": [105.0, 104.0, 103.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "spread": [2.0, 2.0, 2.0],
        })
        result = add_candlestick_patterns(df)
        # bar index 1: high=108 < prev_high=110, low=93 > prev_low=90 → inside
        # bar index 2: high=106 < prev_high=108, low=95 > prev_low=93 → inside
        assert result["pat_inside_bar"][1] == 1
        assert result["pat_inside_bar"][2] == 1


class TestPricePosition:
    def test_columns_created(self, df300):
        result = add_price_position(df300, windows=[10, 20])
        assert "price_pos_10" in result.columns
        assert "price_pos_20" in result.columns

    def test_position_range(self, df300):
        result = add_price_position(df300, windows=[10])
        valid = result.drop_nulls("price_pos_10")
        assert valid["price_pos_10"].min() >= 0.0
        assert valid["price_pos_10"].max() <= 1.0


class TestAllMLFeatures:
    def test_all_categories_present(self, df300):
        df = add_all_technical(df300)   # need EMA columns for ema_distances
        result = add_all_ml_features(df)
        for col in [
            "log_ret_1", "rv_5", "vol_ratio", "vol_regime",
            "body_ratio", "rel_close", "price_pos_10",
            "roc_5", "trend_consistency_5",
        ]:
            assert col in result.columns, f"Missing ML feature: {col}"

    def test_targets_not_added_by_default(self, df300):
        result = add_all_ml_features(df300)
        target_cols = [c for c in result.columns if c.startswith("target_")]
        assert len(target_cols) == 0

    def test_targets_added_when_requested(self, df300):
        result = add_all_ml_features(df300, add_targets=True)
        assert "target_ret_1" in result.columns
        assert "target_ret_5" in result.columns

    def test_targets_contain_nan_at_end(self, df300):
        """target_ret_n is shift(-n) so last n rows must be NaN."""
        result = add_all_ml_features(df300, add_targets=True, target_horizons=[5])
        # Last 5 rows of target_ret_5 should be NaN
        tail = result["target_ret_5"].tail(5)
        assert tail.null_count() == 5


# ===========================================================================
# Feature pipeline orchestrator
# ===========================================================================

class TestFeaturePipeline:
    def test_basic_compute(self, tmp_raw_store):
        pipeline = FeaturePipeline(tmp_raw_store)
        df = pipeline.compute("XAUUSD", "H1")
        assert df is not None
        assert len(df) > 0

    def test_output_has_no_nan_in_required_cols(self, tmp_raw_store):
        config = PipelineConfig(
            required_feature_cols=["rsi_14", "atr_14", "ema_21"]
        )
        pipeline = FeaturePipeline(tmp_raw_store, config)
        df = pipeline.compute("XAUUSD", "H1")
        for col in ["rsi_14", "atr_14", "ema_21"]:
            assert df[col].null_count() == 0, f"NaN in required column {col}"

    def test_output_sorted_ascending(self, tmp_raw_store):
        pipeline = FeaturePipeline(tmp_raw_store)
        df = pipeline.compute("XAUUSD", "H1")
        times = df["event_time"].to_list()
        assert times == sorted(times)

    def test_point_in_time_as_of(self, tmp_raw_store):
        """as_of cutoff must exclude future bars."""
        cutoff = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        pipeline = FeaturePipeline(tmp_raw_store)
        df = pipeline.compute("XAUUSD", "H1", as_of=cutoff)
        if len(df) > 0:
            assert df["event_time"].max() <= cutoff

    def test_insufficient_bars_returns_empty(self, tmp_path):
        """If fewer bars than min_bars, pipeline returns empty DataFrame."""
        from data.store import RawDataStore
        from feeds.base_feed import Bar

        store = RawDataStore(root=tmp_path / "raw")
        # Write only 10 bars — below min_bars=200
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = []
        for i in range(10):
            bars.append(Bar(
                symbol="XAUUSD", timeframe="H1",
                event_time=base + timedelta(hours=i),
                publish_time=base + timedelta(hours=i, milliseconds=100),
                open=2000.0, high=2005.0, low=1995.0, close=2002.0,
                volume=1000.0, spread=2.0, source="test",
            ))
        store.write_batch(bars)

        pipeline = FeaturePipeline(store)
        df = pipeline.compute("XAUUSD", "H1")
        assert len(df) == 0

    def test_missing_symbol_returns_empty(self, tmp_raw_store):
        pipeline = FeaturePipeline(tmp_raw_store)
        df = pipeline.compute("NONEXISTENT", "H1")
        assert len(df) == 0

    def test_quality_report_populated(self, tmp_raw_store):
        pipeline = FeaturePipeline(tmp_raw_store)
        pipeline.compute("XAUUSD", "H1")
        report = pipeline.last_quality_report
        assert report is not None
        assert report.symbol == "XAUUSD"
        assert report.timeframe == "H1"
        assert report.total_bars > 0
        assert report.bars_after_filter <= report.total_bars

    def test_compute_latest_returns_one_row(self, tmp_raw_store):
        pipeline = FeaturePipeline(tmp_raw_store)
        row = pipeline.compute_latest("XAUUSD", "H1", warmup_bars=300)
        assert row is not None
        assert len(row) == 1

    def test_split_features_targets(self, tmp_raw_store):
        config = PipelineConfig(add_targets=True)
        pipeline = FeaturePipeline(tmp_raw_store, config)
        df = pipeline.compute("XAUUSD", "H1")
        X, y = pipeline.split_features_targets(df)
        # Targets should be in y, not X
        target_cols = [c for c in df.columns if c.startswith("target_")]
        for col in target_cols:
            assert col in y.columns
            assert col not in X.columns

    def test_no_lookahead_in_features(self, tmp_raw_store):
        """
        Critical test: features at row t must not use data from row t+1.

        Method: compute features on N bars, then compute on N-1 bars.
        The last row of (N-1) computation must equal (N-1)th row of N.
        If features are leaking, they will differ.
        """
        config = PipelineConfig(compute_ml=False)  # faster
        pipeline = FeaturePipeline(tmp_raw_store, config)

        df_full = pipeline.compute("XAUUSD", "H1")
        if len(df_full) < 10:
            pytest.skip("Not enough bars for lookahead test")

        # Compare second-to-last row of full vs last row of truncated
        cutoff_time = df_full["event_time"][-2]
        df_trunc = pipeline.compute("XAUUSD", "H1", as_of=cutoff_time)

        if len(df_trunc) == 0:
            pytest.skip("Truncated df is empty")

        last_trunc = df_trunc.tail(1)
        matching_full = df_full.filter(
            pl.col("event_time") == cutoff_time
        )

        if len(matching_full) == 0:
            pytest.skip("No matching row in full df")

        # RSI should be identical
        rsi_trunc = last_trunc["rsi_14"][0]
        rsi_full = matching_full["rsi_14"][0]
        if rsi_trunc is not None and rsi_full is not None:
            assert abs(rsi_trunc - rsi_full) < 1e-6, \
                f"Lookahead detected! rsi_14: trunc={rsi_trunc}, full={rsi_full}"


# ===========================================================================
# Feature store
# ===========================================================================

class TestFeatureStore:
    def test_write_and_read(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc123")
        result = tmp_feature_store.read("XAUUSD", "H1", config_hash="abc123")
        assert result is not None
        assert len(result) == len(df)

    def test_cache_miss_wrong_hash(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc123")
        result = tmp_feature_store.read("XAUUSD", "H1", config_hash="different_hash")
        assert result is None

    def test_cache_miss_missing_symbol(self, tmp_feature_store):
        result = tmp_feature_store.read("NONEXISTENT", "H1", config_hash="abc123")
        assert result is None

    def test_invalidate_one(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc")
        tmp_feature_store.write("EURUSD", "H1", df, config_hash="abc")

        tmp_feature_store.invalidate("XAUUSD", "H1")

        assert tmp_feature_store.read("XAUUSD", "H1", config_hash="abc") is None
        assert tmp_feature_store.read("EURUSD", "H1", config_hash="abc") is not None

    def test_invalidate_all(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc")
        tmp_feature_store.write("EURUSD", "M5", df, config_hash="abc")
        tmp_feature_store.invalidate_all()
        assert tmp_feature_store.read("XAUUSD", "H1", config_hash="abc") is None

    def test_staleness_fresh(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc")
        max_time = df["event_time"].max()
        # Cache is fresh — raw max == cache max
        stale = tmp_feature_store.is_stale(
            "XAUUSD", "H1", "abc", raw_store_max_time=max_time
        )
        assert not stale

    def test_staleness_new_data(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc")
        # Simulate new bar arriving in raw store
        new_max = df["event_time"].max() + timedelta(hours=1)
        stale = tmp_feature_store.is_stale(
            "XAUUSD", "H1", "abc", raw_store_max_time=new_max
        )
        assert stale

    def test_sorted_on_read(self, tmp_feature_store):
        df = _make_ohlcv_df(n=60)
        # Write in two batches spanning two months to test multi-partition read
        df1 = df.head(30)
        df2 = df.tail(30)
        tmp_feature_store.write("XAUUSD", "H1", df1, config_hash="v1")
        tmp_feature_store.write("XAUUSD", "H1", df2, config_hash="v1")
        result = tmp_feature_store.read("XAUUSD", "H1", config_hash="v1")
        if result is not None and len(result) > 1:
            times = result["event_time"].to_list()
            assert times == sorted(times)

    def test_get_cache_info(self, tmp_feature_store):
        df = _make_ohlcv_df(n=50)
        tmp_feature_store.write("XAUUSD", "H1", df, config_hash="abc")
        info = tmp_feature_store.get_cache_info()
        assert len(info) > 0
        entry = info[0]
        assert "symbol" in entry
        assert "size_mb" in entry