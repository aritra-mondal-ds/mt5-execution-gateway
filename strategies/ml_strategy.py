"""
strategies/ml_strategy.py
-------------------------
Production Machine Learning Strategy Bridge.

Consumes serialized model artifacts (LightGBM booster or XGBoost classifier)
trained offline in the research platform (TraderBot).

Execution Lifecycle:
  1. Loads model weights (model.txt or model.json) and metadata (meta.json).
  2. Receives live feature dictionary from the FeaturePipeline on each bar.
  3. Validates required stationary feature alignment.
  4. Runs sub-millisecond inference to compute class probabilities:
     P(SHORT), P(HOLD), P(LONG).
  5. Emits a typed Signal with confidence strength if max(P(LONG), P(SHORT)) >= threshold.
  6. Calculates ATR-derived dynamic Stop Loss and Take Profit levels for the RiskEngine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, Direction, Signal


class MLStrategy(BaseStrategy):
    """
    Quantitative Machine Learning Strategy.

    Evaluates live feature vectors against an offline-trained gradient boosted
    decision tree (LightGBM / XGBoost).
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model: Any = None,
        feature_cols: list[str] | None = None,
        model_path: str | Path | None = None,
        meta_path: str | Path | None = None,
        confidence_threshold: float = 0.55,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
        exit_threshold: float = 0.40,
        max_bars_in_trade: int | None = None,
        atr_multiplier: float = 1.5,
        tp_multiplier: float = 2.5,
        cooldown_bars: int = 1,
        warmup_bars: int = 100,
        strategy_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        sid = strategy_id or f"ml_{symbol.lower()}_{timeframe.lower()}"
        super().__init__(
            strategy_id=sid,
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=warmup_bars,
        )
        self.long_threshold = long_threshold if long_threshold is not None else confidence_threshold
        self.short_threshold = short_threshold if short_threshold is not None else confidence_threshold
        self.confidence_threshold = confidence_threshold
        self.exit_threshold = exit_threshold
        self.max_bars_in_trade = max_bars_in_trade
        self.atr_multiplier = atr_multiplier
        self.tp_multiplier = tp_multiplier
        self.cooldown_bars = cooldown_bars

        self._cooldown_counter: int = 0
        self._last_proba: tuple[float, float] | None = None  # (p_long, p_short)
        self._model: Any = model
        self._feature_cols: list[str] = list(feature_cols) if feature_cols is not None else []
        self._model_type: str = "custom" if model is not None else "none"

        # Attempt to load model and metadata from disk if not directly supplied
        if self._model is None and feature_cols is None:
            self._resolve_and_load_model(model_path, meta_path)
        elif model_path is not None:
            self._resolve_and_load_model(model_path, meta_path)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and bool(self._feature_cols)

    # ------------------------------------------------------------------
    # Model Loading & Resolution
    # ------------------------------------------------------------------

    def _resolve_and_load_model(
        self,
        model_path: str | Path | None,
        meta_path: str | Path | None,
    ) -> None:
        """Locates and loads model weights and feature metadata."""
        root = Path(__file__).resolve().parents[1]
        sibling_research_root = root.parent / "TradingBot" / "crypto" / "models"

        candidates = []
        if model_path:
            candidates.append(Path(model_path))

        # Check local and sibling LightGBM models
        candidates.extend([
            root / "models" / "registry" / "lightgbm" / "model.txt",
            root / "models" / "lightgbm" / "model.txt",
            sibling_research_root / "lightgbm" / "model.txt",
        ])

        # Check local and sibling XGBoost models (prefer v2 anti-overfitting)
        candidates.extend([
            root / "models" / "registry" / "xgboost_v2" / "model.json",
            root / "models" / "xgboost_v2" / "model.json",
            sibling_research_root / "xgboost_v2" / "model.json",
            root / "models" / "registry" / "xgboost" / "model.json",
            root / "models" / "xgboost" / "model.json",
            sibling_research_root / "xgboost" / "model.json",
        ])

        target_model = None
        for c in candidates:
            if c.exists():
                target_model = c
                break

        if target_model is None:
            logger.warning(
                f"[{self.strategy_id}] No serialized ML model found at search paths. "
                f"Strategy will operate in passive observation mode."
            )
            return

        # Resolve metadata
        target_meta = Path(meta_path) if meta_path else target_model.parent / "meta.json"
        if not target_meta.exists():
            pkl_meta = target_model.parent / "meta.pkl"
            if pkl_meta.exists():
                import pickle
                with open(pkl_meta, "rb") as f:
                    meta = pickle.load(f)
                if not self._feature_cols:
                    self._feature_cols = meta.get("feature_cols", [])
                self.confidence_threshold = meta.get("threshold", self.confidence_threshold)
            else:
                logger.warning(f"[{self.strategy_id}] Metadata not found at {target_meta}")
        else:
            try:
                meta = json.loads(target_meta.read_text())
                if not self._feature_cols:
                    self._feature_cols = meta.get("feature_cols", [])
                self.confidence_threshold = meta.get("threshold", self.confidence_threshold)
            except Exception as e:
                logger.error(f"[{self.strategy_id}] Error reading metadata: {e}")

        # Load weights
        try:
            if target_model.suffix in (".txt", ""):
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(target_model))
                self._model_type = "lightgbm"
                logger.info(
                    f"[{self.strategy_id}] LightGBM model loaded ← {target_model.name} "
                    f"({len(self._feature_cols)} features, threshold={self.confidence_threshold:.2f})"
                )
            elif target_model.suffix == ".json":
                import xgboost as xgb
                self._model = xgb.XGBClassifier()
                self._model.load_model(str(target_model))
                self._model_type = "xgboost"
                logger.info(
                    f"[{self.strategy_id}] XGBoost model loaded ← {target_model.name} "
                    f"({len(self._feature_cols)} features, threshold={self.confidence_threshold:.2f})"
                )
        except Exception as e:
            logger.error(f"[{self.strategy_id}] Failed to load model from {target_model}: {e}")
            self._model = None

    # ------------------------------------------------------------------
    # Core Signal Generation
    # ------------------------------------------------------------------

    def on_bar(self, row: dict) -> Signal | None:
        if not self.is_ready:
            return None

        # Check in-trade position management first
        in_position = self._state.current_position != Direction.FLAT
        if in_position:
            # Check maximum holding period
            if (
                self.max_bars_in_trade is not None
                and self._state.bars_in_trade >= self.max_bars_in_trade
            ):
                return self._create_exit_signal(row, "max_holding_period")

        # Check cooldown counter for new entries
        if not in_position and self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        # Extract features
        missing_feature = any(
            col not in row or row[col] is None or (isinstance(row[col], float) and np.isnan(row[col]))
            for col in self._feature_cols
        )

        if missing_feature:
            if self._last_proba is None:
                return None
            p_long, p_short = self._last_proba
            p_hold = max(0.0, 1.0 - (p_long + p_short))
        else:
            x = np.array([[float(row[c]) for c in self._feature_cols]], dtype=np.float32)
            try:
                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba(x)[0]
                elif hasattr(self._model, "predict"):
                    res = self._model.predict(x)
                    proba = res[0] if (
                        (hasattr(res, "ndim") and res.ndim > 1)
                        or (isinstance(res, (list, tuple)) and isinstance(res[0], (list, tuple, np.ndarray)))
                    ) else res
                else:
                    return None
            except Exception as e:
                logger.debug(f"[{self.strategy_id}] Inference error: {e}")
                return None

            if len(proba) == 3:
                p_short = float(proba[0])
                p_hold  = float(proba[1])
                p_long  = float(proba[2])
            elif len(proba) == 2:
                p_short = float(proba[0])
                p_hold  = 0.0
                p_long  = float(proba[1])
            else:
                p_short, p_hold, p_long = 0.0, 1.0, 0.0

            self._last_proba = (p_long, p_short)

        # Check confidence degradation for existing positions
        if in_position:
            if (
                self._state.current_position == Direction.LONG
                and p_long < self.exit_threshold
            ) or (
                self._state.current_position == Direction.SHORT
                and p_short < self.exit_threshold
            ):
                return self._create_exit_signal(row, "confidence_dropped")
            return None

        # New entry decision logic
        direction: Direction | None = None
        strength: float = 0.0

        if p_long >= self.long_threshold and p_long > p_short:
            direction = Direction.LONG
            strength = p_long
        elif p_short >= self.short_threshold and p_short > p_long:
            direction = Direction.SHORT
            strength = p_short

        if direction is None:
            return None

        close = float(row.get("close", 0.0))
        atr = float(row.get("atr_14", 0.0))

        bar_time = row.get("event_time")
        if bar_time is None:
            bar_time = datetime.now(tz=timezone.utc)
        elif bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        stop_dist = self.atr_multiplier * atr if atr > 0 else (close * 0.015 if close > 0 else 1.0)
        tp_dist = self.tp_multiplier * atr if atr > 0 else (close * 0.025 if close > 0 else 2.0)

        if direction == Direction.LONG:
            sl = close - stop_dist if close > 0 else None
            tp = close + tp_dist if close > 0 else None
        else:
            sl = close + stop_dist if close > 0 else None
            tp = close - tp_dist if close > 0 else None

        self._cooldown_counter = self.cooldown_bars

        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=direction,
            strength=strength,
            strategy_id=self.strategy_id,
            bar_time=bar_time,
            stop_loss=round(sl, 5) if sl is not None else None,
            take_profit=round(tp, 5) if tp is not None else None,
            atr_at_signal=atr,
            meta={
                "p_short": round(p_short, 4),
                "p_hold": round(p_hold, 4),
                "p_long": round(p_long, 4),
                "model_type": self._model_type,
            },
        )

    def _create_exit_signal(self, row: dict, exit_reason: str) -> Signal:
        bar_time = row.get("event_time")
        if bar_time is None:
            bar_time = datetime.now(tz=timezone.utc)
        elif bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=Direction.FLAT,
            strength=1.0,
            strategy_id=self.strategy_id,
            bar_time=bar_time,
            meta={"exit_reason": exit_reason},
        )

    def reset(self) -> None:
        self._cooldown_counter = 0
        self._last_proba = None
        self._bar_count = 0
