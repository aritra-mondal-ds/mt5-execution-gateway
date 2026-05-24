"""
tests/test_models.py
---------------------
Tests for the models layer.

Coverage:
  - ModelRegistry: save, load, versioning, list, delete
  - make_labels: binary and ternary label construction
  - walk_forward_splits: correct temporal ordering, no leakage
  - ModelTrainer: trains on synthetic data (no GPU required for tests)
  - ModelEvaluator: evaluation metrics on known-answer data

LightGBM is required. GPU is NOT required for tests
(tests use cpu device override).

Run with:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import math
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.registry import ModelRegistry
from models.train import make_labels, walk_forward_splits, TrainConfig, ModelTrainer
from models.evaluate import ModelEvaluator


# ===========================================================================
# Helpers
# ===========================================================================

def _synthetic_df(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Synthetic feature DataFrame for training tests."""
    rng = np.random.default_rng(seed)
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    prices = [2000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0003, 0.005)))

    prices = np.array(prices)
    closes = prices
    opens = prices * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.003, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.003, n))

    event_times = [base + timedelta(hours=i) for i in range(n)]

    df = pl.DataFrame({
        "symbol": ["XAUUSD"] * n,
        "timeframe": ["H1"] * n,
        "event_time": event_times,
        "publish_time": event_times,
        "open": opens.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "close": closes.tolist(),
        "volume": rng.uniform(5000, 20000, n).tolist(),
        "spread": [0.30] * n,
        "source": ["test"] * n,
        # Synthetic features (normally from FeaturePipeline)
        "rsi_14": rng.uniform(20, 80, n).tolist(),
        "atr_14": rng.uniform(10, 30, n).tolist(),
        "ema_9": (prices * (1 + rng.normal(0, 0.002, n))).tolist(),
        "ema_21": (prices * (1 + rng.normal(0, 0.001, n))).tolist(),
        "adx_14": rng.uniform(10, 50, n).tolist(),
        "macd_line": rng.normal(0, 5, n).tolist(),
        "log_ret_1": np.diff(np.log(prices), prepend=np.log(prices[0])).tolist(),
        "vol_ratio": rng.uniform(0.5, 2.0, n).tolist(),
    })
    return df


class MockBooster:
    """
    Module-level mock that mimics LightGBM's predict interface.
    Must be at module level — pickle requires a stable import path.
    """
    def predict(self, X):
        rng = np.random.default_rng(0)
        return rng.uniform(0.3, 0.7, len(X))

    def feature_importance(self, importance_type="gain"):
        return np.ones(5)


def _mock_model() -> MockBooster:
    return MockBooster()


# ===========================================================================
# ModelRegistry
# ===========================================================================

class TestModelRegistry:
    def test_save_and_load(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        model = _mock_model()
        feature_cols = ["rsi_14", "atr_14", "ema_9"]

        reg.save(model, feature_cols, "XAUUSD", "H1", "lgbm",
                 performance={"auc": 0.62})
        bundle = reg.load("XAUUSD", "H1", "lgbm")

        assert "model" in bundle
        assert bundle["feature_cols"] == feature_cols
        assert bundle["metadata"]["symbol"] == "XAUUSD"

    def test_auto_version_increments(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        model = _mock_model()
        cols = ["rsi_14"]

        reg.save(model, cols, "XAUUSD", "H1", "lgbm")
        reg.save(model, cols, "XAUUSD", "H1", "lgbm")

        v = reg.get_latest_version("XAUUSD", "H1", "lgbm")
        assert v == 2

    def test_load_specific_version(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        model = _mock_model()
        cols = ["rsi_14"]

        reg.save(model, cols, "XAUUSD", "H1", "lgbm",
                 extra_metadata={"tag": "v1"})
        reg.save(model, cols, "XAUUSD", "H1", "lgbm",
                 extra_metadata={"tag": "v2"})

        bundle = reg.load("XAUUSD", "H1", "lgbm", version=1)
        assert bundle["metadata"]["tag"] == "v1"

    def test_load_nonexistent_raises(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.load("NONEXISTENT", "H1", "lgbm")

    def test_list_models(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        model = _mock_model()
        reg.save(model, ["f1"], "XAUUSD", "H1", "lgbm")
        reg.save(model, ["f1"], "EURUSD", "H1", "lgbm")

        all_models = reg.list_models()
        assert len(all_models) == 2

        filtered = reg.list_models(symbol="XAUUSD")
        assert len(filtered) == 1
        assert filtered[0]["symbol"] == "XAUUSD"

    def test_delete_model(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        reg.save(_mock_model(), ["f1"], "XAUUSD", "H1", "lgbm")
        deleted = reg.delete("XAUUSD", "H1", "lgbm", version=1)
        assert deleted
        assert reg.get_latest_version("XAUUSD", "H1", "lgbm") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        deleted = reg.delete("XAUUSD", "H1", "lgbm", version=99)
        assert not deleted

    def test_metadata_stored_correctly(self, tmp_path):
        reg = ModelRegistry(root=tmp_path)
        perf = {"auc": 0.65, "sharpe": 1.2}
        reg.save(_mock_model(), ["f1", "f2"], "XAUUSD", "H1", "lgbm",
                 performance=perf)
        bundle = reg.load("XAUUSD", "H1", "lgbm")
        assert bundle["metadata"]["performance"]["auc"] == pytest.approx(0.65)
        assert bundle["metadata"]["n_features"] == 2


# ===========================================================================
# Label construction
# ===========================================================================

class TestMakeLabels:
    def test_binary_labels_are_0_or_1(self):
        df = _synthetic_df(100)
        labels = make_labels(df, horizon=5, threshold=0.002, label_type="binary")
        valid = labels.drop_nulls()
        assert set(valid.to_list()).issubset({0, 1})

    def test_ternary_labels_are_0_1_or_2(self):
        df = _synthetic_df(100)
        labels = make_labels(df, horizon=5, threshold=0.002, label_type="ternary")
        valid = labels.drop_nulls()
        assert set(valid.to_list()).issubset({0, 1, 2})

    def test_last_n_labels_are_null(self):
        df = _synthetic_df(100)
        horizon = 5
        labels = make_labels(df, horizon=horizon, threshold=0.002)
        # Last `horizon` rows should be null (no forward data)
        tail_nulls = labels.tail(horizon).null_count()
        assert tail_nulls == horizon

    def test_label_direction_matches_return(self):
        """If forward return > threshold, label must be 1."""
        # Construct a price series guaranteed to go up
        prices = [float(i + 1000) for i in range(20)]
        df = pl.DataFrame({
            "close": prices,
            "event_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 20,
        })
        labels = make_labels(df, horizon=1, threshold=0.0001, label_type="binary")
        # All forward returns are positive and large → all should be 1
        valid = labels.drop_nulls()
        assert all(v == 1 for v in valid.to_list())

    def test_uses_existing_target_col_if_present(self):
        """If target_ret_5 already exists, don't recompute."""
        df = _synthetic_df(50)
        # Add a pre-computed target column with all 1s
        df = df.with_columns(pl.lit(1).alias("target_ret_5"))
        labels = make_labels(df, horizon=5, threshold=0.002, label_type="binary")
        # Should use the existing column → all 1s
        valid = labels.drop_nulls()
        assert all(v == 1 for v in valid.to_list())


# ===========================================================================
# Walk-forward splits
# ===========================================================================

class TestWalkForwardSplits:
    def test_returns_correct_number_of_splits(self):
        splits = walk_forward_splits(
            n_samples=3000, n_splits=5,
            train_size=500, val_size=200,
        )
        assert len(splits) <= 5

    def test_no_overlap_between_train_and_val(self):
        splits = walk_forward_splits(
            n_samples=2000, n_splits=3,
            train_size=500, val_size=200,
        )
        for train_idx, val_idx in splits:
            train_set = set(train_idx.tolist())
            val_set = set(val_idx.tolist())
            assert train_set.isdisjoint(val_set), "Train and val overlap!"

    def test_val_always_after_train(self):
        splits = walk_forward_splits(
            n_samples=2000, n_splits=3,
            train_size=500, val_size=200,
        )
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min(), \
                "Validation indices precede training indices!"

    def test_expanding_window_grows(self):
        splits = walk_forward_splits(
            n_samples=3000, n_splits=3,
            train_size=500, val_size=200,
            expanding=True,
        )
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes), "Expanding window should grow"

    def test_rolling_window_fixed_size(self):
        splits = walk_forward_splits(
            n_samples=3000, n_splits=3,
            train_size=500, val_size=200,
            expanding=False,
        )
        # Rolling: training size should be approximately constant
        sizes = [len(train) for train, _ in splits]
        if len(sizes) > 1:
            # Allow small variance at boundaries
            assert max(sizes) - min(sizes) <= 10

    def test_insufficient_data_returns_empty(self):
        splits = walk_forward_splits(
            n_samples=100, n_splits=5,
            train_size=500, val_size=200,
        )
        assert len(splits) == 0


# ===========================================================================
# ModelTrainer (requires LightGBM)
# ===========================================================================

lgbm_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("lightgbm"),
    reason="lightgbm not installed",
)


class TestModelTrainer:
    @lgbm_available
    def test_trainer_produces_model(self, tmp_path):
        """Full training pipeline on synthetic data, CPU only."""
        df = _synthetic_df(n=1500, seed=42)

        config = TrainConfig(
            symbol="XAUUSD",
            timeframe="H1",
            target_horizon=5,
            target_threshold=0.002,
            label_type="binary",
            n_splits=2,
            train_size=400,
            val_size=150,
            use_optuna=False,
            lgbm_params={
                "objective": "binary",
                "metric": "auc",
                "device": "cpu",     # CPU for tests
                "num_leaves": 15,
                "learning_rate": 0.1,
                "n_estimators": 50,  # fast
                "early_stopping_rounds": 10,
                "verbosity": -1,
            },
        )

        trainer = ModelTrainer(config)
        result = trainer.train(df)

        assert "model" in result
        assert "feature_cols" in result
        assert "cv_metrics" in result
        assert len(result["feature_cols"]) > 0
        assert 0 <= result["cv_metrics"]["mean_auc"] <= 1

    @lgbm_available
    def test_trained_model_predicts_probabilities(self):
        df = _synthetic_df(n=1000, seed=7)
        config = TrainConfig(
            symbol="XAUUSD", timeframe="H1",
            n_splits=2, train_size=300, val_size=100,
            lgbm_params={
                "objective": "binary", "metric": "auc",
                "device": "cpu", "num_leaves": 15,
                "n_estimators": 30, "early_stopping_rounds": 5,
                "verbosity": -1,
            },
        )
        result = ModelTrainer(config).train(df)
        model = result["model"]
        feature_cols = result["feature_cols"]

        X = df.select(feature_cols).to_numpy().astype(np.float32)
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert all(0.0 <= p <= 1.0 for p in preds)


# ===========================================================================
# ModelEvaluator
# ===========================================================================

class TestModelEvaluator:
    def _bundle(self, feature_cols):
        return {
            "model": _mock_model(),
            "feature_cols": feature_cols,
            "metadata": {
                "symbol": "XAUUSD",
                "timeframe": "H1",
                "model_type": "lgbm",
            },
        }

    def test_evaluator_returns_report(self):
        df = _synthetic_df(200)
        feature_cols = ["rsi_14", "atr_14", "ema_9", "adx_14", "macd_line"]
        bundle = self._bundle(feature_cols)
        evaluator = ModelEvaluator(bundle)
        report = evaluator.evaluate(df, horizon=5, threshold=0.002)
        assert report is not None
        assert report.n_samples > 0
        assert 0 <= report.auc <= 1

    def test_accuracy_between_0_and_1(self):
        df = _synthetic_df(200)
        cols = ["rsi_14", "atr_14", "ema_9", "adx_14", "macd_line"]
        evaluator = ModelEvaluator(self._bundle(cols))
        report = evaluator.evaluate(df, horizon=5)
        assert 0.0 <= report.accuracy <= 1.0

    def test_signal_count_bounded(self):
        df = _synthetic_df(200)
        cols = ["rsi_14", "atr_14", "ema_9", "adx_14", "macd_line"]
        evaluator = ModelEvaluator(self._bundle(cols))
        report = evaluator.evaluate(df, horizon=5, signal_prob_threshold=0.60)
        assert report.n_signals >= 0
        assert report.n_signals <= report.n_samples

    def test_print_summary_runs(self):
        from io import StringIO
        import sys
        df = _synthetic_df(200)
        cols = ["rsi_14", "atr_14", "ema_9", "adx_14", "macd_line"]
        evaluator = ModelEvaluator(self._bundle(cols))
        report = evaluator.evaluate(df)

        captured = StringIO()
        sys.stdout = captured
        report.print_summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "AUC" in output
        assert "XAUUSD" in output

    def test_missing_features_handled_gracefully(self):
        """Evaluator should warn but not crash on missing feature cols."""
        df = _synthetic_df(200)
        # Bundle expects a col that doesn't exist in df
        cols = ["rsi_14", "nonexistent_feature_col"]
        evaluator = ModelEvaluator(self._bundle(cols))
        report = evaluator.evaluate(df, horizon=5)
        assert report is not None  # no crash