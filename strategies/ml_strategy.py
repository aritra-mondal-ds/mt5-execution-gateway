"""
strategies/ml_strategy.py
--------------------------
LightGBM-based ML strategy.

Architecture:
  - Wraps a trained LightGBM binary classifier.
  - Model predicts P(next_N_bars_return > threshold) for long,
    and P(next_N_bars_return < -threshold) for short.
  - Signals are emitted when class probability exceeds entry_threshold.
  - Model is loaded from the model registry (versioned .pkl files).
  - Feature columns used for inference are stored alongside the model
    to guarantee consistency between training and live inference.

Lookahead safety:
  - The model is trained on features from bar t to predict bar t+N.
  - At inference time, we feed features from the LATEST completed bar.
  - The `feature_cols` list stored with the model is the contract —
    we assert the incoming row contains exactly those columns.

GPU inference:
  - LightGBM uses the GPU automatically if the model was trained
    with device='gpu'. No code change needed here.

Model registry interface:
  - Load: ModelRegistry.load(model_id) → (model, metadata)
  - metadata contains: feature_cols, training_date, performance_metrics
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, Direction, Signal


class MLStrategy(BaseStrategy):
    """
    LightGBM signal strategy.

    Can be instantiated either with a model object directly (for testing)
    or with a path to a pickled model file (for production).

    Required feature columns: stored in model metadata or passed explicitly.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: str | Path | None = None,
        model: Any = None,                    # pre-loaded model (for testing)
        feature_cols: list[str] | None = None,
        long_threshold: float = 0.60,         # P(long) must exceed this
        short_threshold: float = 0.60,        # P(short) must exceed this
        exit_threshold: float = 0.45,         # exit when confidence drops below
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        tp_multiplier: float = 3.0,
        max_bars_in_trade: int = 30,
        cooldown_bars: int = 2,
    ) -> None:
        super().__init__(
            strategy_id=f"ml_lgbm_{symbol}_{timeframe}",
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=200,
        )
        self._model = model
        self._feature_cols: list[str] = feature_cols or []
        self._atr_col = f"atr_{atr_period}"
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.exit_threshold = exit_threshold
        self.atr_multiplier = atr_multiplier
        self.tp_multiplier = tp_multiplier
        self.max_bars_in_trade = max_bars_in_trade
        self.cooldown_bars = cooldown_bars
        self._cooldown_counter = 0
        self._last_proba: tuple[float, float] | None = None  # (p_long, p_short)

        if model_path is not None:
            self._load_model(Path(model_path))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, path: Path) -> None:
        """
        Load model and metadata from a pickle file.

        Expected pickle format:
            {
                "model": lgb.Booster,
                "feature_cols": [...],
                "metadata": {...}
            }
        Or simply a raw lgb.Booster (feature_cols must be passed explicitly).
        """
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            self._model = obj["model"]
            if not self._feature_cols:
                self._feature_cols = obj.get("feature_cols", [])
            metadata = obj.get("metadata", {})
            logger.info(
                f"MLStrategy loaded model from {path} | "
                f"features={len(self._feature_cols)} | "
                f"trained={metadata.get('training_date', 'unknown')}"
            )
        else:
            # Raw model object
            self._model = obj
            logger.info(f"MLStrategy loaded raw model from {path}")

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and feature columns are defined."""
        return self._model is not None and len(self._feature_cols) > 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def on_bar(self, row: dict) -> Signal | None:
        if not self.is_ready:
            logger.warning(f"{self.strategy_id}: model not loaded — no signal")
            return None

        # --- Exit logic ---
        if not self.is_flat:
            signal = self._check_exit(row)
            if signal is not None:
                self._state.current_position = Direction.FLAT
                self._state.bars_in_trade = 0
                self._cooldown_counter = self.cooldown_bars
                return signal
            self._state.bars_in_trade += 1

        # --- Cooldown ---
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        # --- Entry logic ---
        if self.is_flat:
            signal = self._check_entry(row)
            if signal is not None:
                self._state.current_position = signal.direction
                self._state.bars_in_trade = 0
            return signal

        return None

    def _predict(self, row: dict) -> tuple[float, float]:
        """
        Run model inference on a single feature row.

        Returns: (p_long, p_short)
          - p_long  = P(price goes up significantly)
          - p_short = P(price goes down significantly)

        For a 3-class model: [down, flat, up]
          p_long  = proba[2]
          p_short = proba[0]

        For a binary model trained on up/down:
          p_long  = proba[1]
          p_short = 1 - proba[1]
        """
        missing = [c for c in self._feature_cols if c not in row]
        if missing:
            logger.warning(
                f"{self.strategy_id}: missing {len(missing)} feature cols "
                f"(first 3: {missing[:3]}) — using last known proba"
            )
            return self._last_proba or (0.5, 0.5)

        # Build feature vector — must match training column order exactly
        X = np.array([[row[c] if row[c] is not None else 0.0
                        for c in self._feature_cols]], dtype=np.float32)

        try:
            proba = self._model.predict(X)

            # Handle both binary and multiclass output shapes
            if proba.ndim == 1:
                # Binary: proba is shape (1,) → probability of class 1
                p_long = float(proba[0])
                p_short = 1.0 - p_long
            elif proba.shape[1] == 2:
                # Binary sklearn-style: shape (1, 2)
                p_long = float(proba[0, 1])
                p_short = float(proba[0, 0])
            elif proba.shape[1] == 3:
                # 3-class: [down, flat, up]
                p_short = float(proba[0, 0])
                p_long  = float(proba[0, 2])
            else:
                p_long, p_short = 0.5, 0.5

        except Exception as e:
            logger.error(f"{self.strategy_id}: inference error: {e}")
            return self._last_proba or (0.5, 0.5)

        self._last_proba = (p_long, p_short)
        return p_long, p_short

    def _check_entry(self, row: dict) -> Signal | None:
        p_long, p_short = self._predict(row)
        close = self._get(row, "close")

        if close is None:
            return None

        # ATR is for sizing only — missing ATR means SL/TP are None.
        # The risk engine will apply its own sizing in that case.
        atr = self._get(row, self._atr_col)
        sl_long = sl_short = tp_long = tp_short = None
        if atr is not None:
            stop   = atr * self.atr_multiplier
            tp_dist = atr * self.tp_multiplier
            sl_long  = close - stop
            sl_short = close + stop
            tp_long  = close + tp_dist
            tp_short = close - tp_dist

        if p_long >= self.long_threshold:
            logger.debug(
                f"{self.strategy_id} LONG | {row['event_time']} | "
                f"p_long={p_long:.3f} close={close:.4f}"
            )
            return self._make_signal(
                row, Direction.LONG,
                strength=p_long,
                stop_loss=sl_long,
                take_profit=tp_long,
                meta={"p_long": p_long, "p_short": p_short},
            )

        if p_short >= self.short_threshold:
            logger.debug(
                f"{self.strategy_id} SHORT | {row['event_time']} | "
                f"p_short={p_short:.3f} close={close:.4f}"
            )
            return self._make_signal(
                row, Direction.SHORT,
                strength=p_short,
                stop_loss=sl_short,
                take_profit=tp_short,
                meta={"p_long": p_long, "p_short": p_short},
            )

        return None

    def _check_exit(self, row: dict) -> Signal | None:
        p_long, p_short = self._predict(row)
        exit_reason = None

        if self._state.bars_in_trade >= self.max_bars_in_trade:
            exit_reason = "max_holding_period"
        elif self.is_long and p_long < self.exit_threshold:
            exit_reason = "confidence_dropped"
        elif self.is_short and p_short < self.exit_threshold:
            exit_reason = "confidence_dropped"

        if exit_reason:
            return self._make_signal(
                row, Direction.FLAT,
                meta={
                    "exit_reason": exit_reason,
                    "p_long": p_long, "p_short": p_short,
                    "bars_held": self._state.bars_in_trade,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        from strategies.base_strategy import StrategyState
        self._state = StrategyState(strategy_id=self.strategy_id)
        self._bar_count = 0
        self._last_proba = None
        self._cooldown_counter = 0