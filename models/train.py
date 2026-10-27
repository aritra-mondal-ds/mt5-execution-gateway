"""
models/train.py
---------------
LightGBM training pipeline with GPU acceleration and Optuna tuning.

Pipeline:
  1. Load features from FeaturePipeline (point-in-time correct)
  2. Construct labels (forward returns → binary/ternary classification)
  3. Walk-forward cross-validation (no future leakage)
  4. Optuna hyperparameter optimisation (optional)
  5. Train final model on all training data
  6. Save to ModelRegistry

Label construction:
  Binary:   1 if forward_return > threshold else 0
  Ternary:  0=down, 1=flat, 2=up  (used for 3-class MLStrategy)

Walk-forward splits:
  Training uses expanding or rolling window.
  Validation is always the NEXT contiguous period.
  No shuffling — temporal ordering is preserved.

GPU:
  LightGBM uses CUDA automatically when device="gpu" is set.
  Your RTX 3060 12GB handles datasets up to ~500k rows comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None  # type: ignore

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    symbol: str
    timeframe: str
    target_horizon: int = 5          # bars ahead to predict
    target_threshold: float = 0.002  # min return to be labeled "up"/"down"
    label_type: str = "binary"       # "binary" or "ternary"

    # Feature exclusions
    exclude_cols: list[str] = field(default_factory=lambda: [
        "symbol", "timeframe", "event_time", "publish_time",
        "open", "high", "low", "close", "volume", "spread", "source",
    ])

    # Walk-forward settings
    n_splits: int = 5
    train_size: int = 2000           # bars per training fold
    val_size: int = 500              # bars per validation fold
    use_expanding_window: bool = True

    # LightGBM base params
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "device": "gpu",             # uses RTX 3060
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbosity": -1,
    })

    # Optuna tuning
    use_optuna: bool = False
    n_optuna_trials: int = 50
    optuna_timeout_seconds: int = 300


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------

def walk_forward_splits(
    n_samples: int,
    n_splits: int,
    train_size: int,
    val_size: int,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_indices, val_indices) pairs for walk-forward CV.

    All validation sets are strictly after all training sets.
    No data point appears in both train and val for any split.
    """
    splits = []
    step = val_size

    for i in range(n_splits):
        val_end = n_samples - (n_splits - 1 - i) * step
        val_start = val_end - val_size
        if val_start < train_size:
            continue

        if expanding:
            train_start = 0
            train_end = val_start
        else:
            train_end = val_start
            train_start = max(0, train_end - train_size)

        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) >= train_size // 2 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def make_labels(
    df: pl.DataFrame,
    horizon: int,
    threshold: float,
    label_type: str = "binary",
) -> pl.Series:
    """
    Construct target labels from forward returns.

    Binary:   1 if return > threshold else 0
    Ternary:  0 if return < -threshold, 1 if |return| <= threshold, 2 if > threshold
    """
    target_col = f"target_ret_{horizon}"
    if target_col not in df.columns:
        # Compute forward returns if not already present
        forward_ret = (
            pl.col("close").shift(-horizon) / pl.col("close") - 1.0
        ).alias(target_col)
        df = df.with_columns(forward_ret)

    rets = df[target_col]

    if label_type == "binary":
        labels = (rets > threshold).cast(pl.Int32)
    else:  # ternary
        labels = (
            pl.when(rets > threshold).then(2)
              .when(rets < -threshold).then(0)
              .otherwise(1)
              .alias("label")
        )
        labels = df.select(labels)["label"]

    return labels


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains a LightGBM model with walk-forward CV.

    Usage:
        trainer = ModelTrainer(config)
        result = trainer.train(feature_df)
        registry.save(
            model=result["model"],
            feature_cols=result["feature_cols"],
            symbol=config.symbol,
            timeframe=config.timeframe,
            performance=result["cv_metrics"],
        )
    """

    def __init__(self, config: TrainConfig) -> None:
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        self._config = config

    def train(self, df: pl.DataFrame) -> dict:
        """
        Full training pipeline.

        Args:
            df: feature DataFrame from FeaturePipeline
                Must include "close" and "event_time" columns.

        Returns:
            {
                "model"       : trained LightGBM Booster,
                "feature_cols": list of feature column names,
                "cv_metrics"  : dict of cross-validation performance,
                "oof_preds"   : out-of-fold predictions array,
            }
        """
        cfg = self._config
        logger.info(
            f"Training | {cfg.symbol}/{cfg.timeframe} | "
            f"horizon={cfg.target_horizon} | samples={len(df)}"
        )

        # Build feature matrix and labels
        feature_cols = self._get_feature_cols(df)
        labels = make_labels(df, cfg.target_horizon, cfg.target_threshold, cfg.label_type)

        # Drop rows where label is null (last horizon bars)
        valid_mask = labels.is_not_null()
        df_clean = df.filter(valid_mask)
        labels_clean = labels.filter(valid_mask)

        X = df_clean.select(feature_cols).to_numpy().astype(np.float32)
        y = labels_clean.to_numpy().astype(np.int32)

        logger.info(
            f"Dataset | X={X.shape} | "
            f"class_balance={np.mean(y):.3f} | "
            f"features={len(feature_cols)}"
        )

        # Walk-forward cross-validation
        splits = walk_forward_splits(
            n_samples=len(X),
            n_splits=cfg.n_splits,
            train_size=cfg.train_size,
            val_size=cfg.val_size,
            expanding=cfg.use_expanding_window,
        )

        if not splits:
            raise ValueError(
                f"Insufficient data for walk-forward CV. "
                f"Need at least {cfg.train_size + cfg.val_size} samples, "
                f"got {len(X)}."
            )

        cv_aucs = []
        oof_preds = np.full(len(X), np.nan)

        params = dict(cfg.lgbm_params)

        # Optuna tuning on first fold
        if cfg.use_optuna and OPTUNA_AVAILABLE:
            train_idx, val_idx = splits[0]
            params = self._tune_hyperparams(
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                base_params=params,
            )

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = self._fit(X_tr, y_tr, X_val, y_val, params)

            val_preds = model.predict(X_val)
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_val, val_preds)
            except Exception:
                auc = 0.5
            cv_aucs.append(auc)
            oof_preds[val_idx] = val_preds

            logger.info(f"Fold {fold_i + 1}/{len(splits)} | AUC={auc:.4f}")

        mean_auc = float(np.mean(cv_aucs))
        std_auc = float(np.std(cv_aucs))
        logger.info(
            f"CV complete | mean_AUC={mean_auc:.4f} ± {std_auc:.4f}"
        )

        # Train final model on all data
        logger.info("Training final model on full dataset...")
        final_model = self._fit(X, y, None, None, params, final=True)

        return {
            "model": final_model,
            "feature_cols": feature_cols,
            "cv_metrics": {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "fold_aucs": cv_aucs,
                "n_splits": len(splits),
            },
            "oof_preds": oof_preds,
            "params": params,
        }

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        params: dict,
        final: bool = False,
    ) -> Any:
        """Fit one LightGBM model."""
        p = dict(params)
        n_estimators = p.pop("n_estimators", 500)
        early_stopping = p.pop("early_stopping_rounds", 50)
        p.pop("verbosity", None)

        dtrain = lgb.Dataset(
            np.ascontiguousarray(X_train, dtype=np.float32),
            label=np.ascontiguousarray(y_train, dtype=np.float32),
            free_raw_data=False,
        )
        callbacks = [lgb.log_evaluation(period=100)]

        if X_val is not None and not final:
            dval = lgb.Dataset(
                np.ascontiguousarray(X_val, dtype=np.float32),
                label=np.ascontiguousarray(y_val, dtype=np.float32),
                reference=dtrain,
                free_raw_data=False,
            )
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            model = lgb.train(
                p, dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dval],
                callbacks=callbacks,
            )
        else:
            model = lgb.train(
                p, dtrain,
                num_boost_round=n_estimators,
                callbacks=[lgb.log_evaluation(period=200)],
            )

        return model

    def _tune_hyperparams(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        base_params: dict,
    ) -> dict:
        """Optuna hyperparameter search."""
        from sklearn.metrics import roc_auc_score

        def objective(trial):
            params = {
                **base_params,
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }
            model = self._fit(X_train, y_train, X_val, y_val, params)
            preds = model.predict(X_val)
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self._config.n_optuna_trials,
            timeout=self._config.optuna_timeout_seconds,
            show_progress_bar=False,
        )

        best = {**base_params, **study.best_params}
        logger.info(
            f"Optuna best AUC={study.best_value:.4f} | "
            f"params={study.best_params}"
        )
        return best

    def _get_feature_cols(self, df: pl.DataFrame) -> list[str]:
        """
        Select feature columns — exclude raw OHLCV, metadata, and targets.
        """
        exclude = set(self._config.exclude_cols)
        return [
            c for c in df.columns
            if c not in exclude
            and not c.startswith("target_")
            and df[c].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int32, pl.Int64)
        ]