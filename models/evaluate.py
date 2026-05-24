"""
models/evaluate.py
------------------
Out-of-sample model evaluation and signal quality analysis.

Evaluates a trained model on held-out data with:
  1. Classification metrics (AUC, precision, recall, F1)
  2. Signal quality metrics (how well model signals predict actual P&L)
  3. Feature importance (gain-based + permutation)
  4. Calibration check (are model probabilities well-calibrated?)
  5. Regime analysis (does model perform differently in different regimes?)

Usage:
    evaluator = ModelEvaluator(model_bundle, config)
    report = evaluator.evaluate(test_df)
    report.print_summary()
    report.plot_feature_importance()  # requires matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class EvaluationReport:
    symbol: str
    timeframe: str
    model_type: str

    # Classification metrics
    auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_samples: int = 0

    # Signal quality (using probability threshold)
    threshold: float = 0.60
    n_signals: int = 0
    signal_win_rate: float = 0.0
    signal_avg_return: float = 0.0

    # Feature importance (top 20)
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Calibration
    calibration_error: float = 0.0  # expected calibration error

    def print_summary(self) -> None:
        print(f"\n{'─'*50}")
        print(f"  MODEL EVALUATION | {self.symbol}/{self.timeframe}/{self.model_type}")
        print(f"{'─'*50}")
        print(f"  Samples          : {self.n_samples:>10}")
        print(f"  AUC              : {self.auc:>10.4f}")
        print(f"  Accuracy         : {self.accuracy:>10.2%}")
        print(f"  Precision        : {self.precision:>10.2%}")
        print(f"  Recall           : {self.recall:>10.2%}")
        print(f"  F1               : {self.f1:>10.4f}")
        print(f"{'─'*50}")
        print(f"  Signal threshold : {self.threshold:>10.2f}")
        print(f"  Signals emitted  : {self.n_signals:>10}")
        print(f"  Signal win rate  : {self.signal_win_rate:>10.2%}")
        print(f"  Signal avg ret   : {self.signal_avg_return:>10.4f}")
        print(f"{'─'*50}")
        if self.feature_importance:
            print("  Top 10 features:")
            top10 = sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for feat, imp in top10:
                print(f"    {feat:<35} {imp:.4f}")
        print(f"{'─'*50}\n")


class ModelEvaluator:
    """
    Evaluate a trained model on out-of-sample data.

    Usage:
        bundle = registry.load("XAUUSD", "H1", "lgbm")
        evaluator = ModelEvaluator(bundle)
        report = evaluator.evaluate(test_feature_df, horizon=5, threshold=0.002)
    """

    def __init__(self, model_bundle: dict) -> None:
        self._model = model_bundle["model"]
        self._feature_cols = model_bundle["feature_cols"]
        self._metadata = model_bundle.get("metadata", {})

    def evaluate(
        self,
        df: pl.DataFrame,
        horizon: int = 5,
        threshold: float = 0.002,
        signal_prob_threshold: float = 0.60,
        label_type: str = "binary",
    ) -> EvaluationReport:
        """
        Run full OOS evaluation.

        Args:
            df                   : test feature DataFrame (must not overlap training)
            horizon              : forward-return horizon for labels
            threshold            : return threshold for label construction
            signal_prob_threshold: probability above which we consider a signal
        """
        from models.train import make_labels

        report = EvaluationReport(
            symbol=self._metadata.get("symbol", ""),
            timeframe=self._metadata.get("timeframe", ""),
            model_type=self._metadata.get("model_type", "lgbm"),
            threshold=signal_prob_threshold,
        )

        # Build labels
        labels = make_labels(df, horizon, threshold, label_type)
        valid = labels.is_not_null()
        df_clean = df.filter(valid)
        y_true = labels.filter(valid).to_numpy().astype(np.int32)

        # Check features
        missing = [c for c in self._feature_cols if c not in df_clean.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} feature cols in test df: {missing[:5]}")
            avail = [c for c in self._feature_cols if c in df_clean.columns]
        else:
            avail = self._feature_cols

        X = df_clean.select(avail).to_numpy().astype(np.float32)
        report.n_samples = len(X)

        if len(X) == 0:
            logger.warning("No valid samples for evaluation")
            return report

        # Predict
        y_prob = self._model.predict(X)
        y_pred = (y_prob >= 0.5).astype(np.int32)

        # ── Classification metrics ───────────────────────────────────
        report.auc = self._auc(y_true, y_prob)
        report.accuracy = float(np.mean(y_pred == y_true))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        report.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        report.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if report.precision + report.recall > 0:
            report.f1 = (
                2 * report.precision * report.recall /
                (report.precision + report.recall)
            )

        # ── Signal quality ───────────────────────────────────────────
        signal_mask = y_prob >= signal_prob_threshold
        report.n_signals = int(np.sum(signal_mask))

        if report.n_signals > 0:
            # Forward returns for signal bars
            target_col = f"target_ret_{horizon}"
            if target_col in df_clean.columns:
                fwd_rets = df_clean.filter(
                    pl.Series(signal_mask)
                )[target_col].drop_nulls().to_numpy()
                if len(fwd_rets) > 0:
                    report.signal_win_rate = float(np.mean(fwd_rets > 0))
                    report.signal_avg_return = float(np.mean(fwd_rets))

        # ── Feature importance ───────────────────────────────────────
        try:
            importances = self._model.feature_importance(importance_type="gain")
            report.feature_importance = {
                col: float(imp)
                for col, imp in zip(avail, importances)
            }
        except Exception:
            pass

        # ── Calibration ──────────────────────────────────────────────
        report.calibration_error = self._calibration_error(y_true, y_prob)

        logger.info(
            f"Evaluation | {report.symbol}/{report.timeframe} | "
            f"AUC={report.auc:.4f} | signals={report.n_signals} | "
            f"win_rate={report.signal_win_rate:.2%}"
        )

        return report

    @staticmethod
    def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return 0.5

    @staticmethod
    def _calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error (ECE)."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            frac_positives = np.mean(y_true[mask])
            mean_prob = np.mean(y_prob[mask])
            ece += np.abs(frac_positives - mean_prob) * np.sum(mask) / n
        return float(ece)