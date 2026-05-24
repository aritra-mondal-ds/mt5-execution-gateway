"""
models/registry.py
------------------
Versioned model registry for LightGBM and PyTorch models.

Each saved model is a dict containing:
    {
        "model"       : the trained model object,
        "feature_cols": list of feature column names used in training,
        "metadata"    : {
            "training_date", "symbol", "timeframe",
            "n_samples", "performance", "model_type"
        }
    }

File naming: {symbol}_{timeframe}_{model_type}_{version}.pkl
Example    : XAUUSD_H1_lgbm_v003.pkl

The registry always loads the latest version unless a specific
version is requested. Version numbers are zero-padded integers.
"""

from __future__ import annotations

import pickle
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class ModelRegistry:
    """
    Save, load, and list versioned models.

    Usage:
        registry = ModelRegistry(root=settings.data.models_dir)

        # Save
        registry.save(
            model=lgb_booster,
            feature_cols=feature_cols,
            symbol="XAUUSD",
            timeframe="H1",
            model_type="lgbm",
            performance={"sharpe": 1.4, "win_rate": 0.55},
        )

        # Load latest
        bundle = registry.load("XAUUSD", "H1", "lgbm")
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        # Load specific version
        bundle = registry.load("XAUUSD", "H1", "lgbm", version=2)
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        feature_cols: list[str],
        symbol: str,
        timeframe: str,
        model_type: str = "lgbm",
        performance: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        """
        Save a model bundle. Auto-increments version number.
        Returns the path of the saved file.
        """
        version = self._next_version(symbol, timeframe, model_type)
        filename = self._filename(symbol, timeframe, model_type, version)
        path = self._root / filename

        bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
                "version": version,
                "training_date": datetime.now(tz=timezone.utc).isoformat(),
                "n_features": len(feature_cols),
                "performance": performance or {},
                **(extra_metadata or {}),
            },
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(
            f"ModelRegistry saved | {filename} | "
            f"features={len(feature_cols)} | "
            f"performance={performance}"
        )
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        timeframe: str,
        model_type: str = "lgbm",
        version: int | None = None,
    ) -> dict:
        """
        Load a model bundle.

        Args:
            version: specific version to load (None = latest)

        Returns:
            dict with keys: model, feature_cols, metadata
        Raises:
            FileNotFoundError if no model found
        """
        if version is None:
            path = self._latest_path(symbol, timeframe, model_type)
        else:
            path = self._root / self._filename(symbol, timeframe, model_type, version)

        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No model found for {symbol}/{timeframe}/{model_type} "
                f"(version={version})"
            )

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        logger.info(
            f"ModelRegistry loaded | {path.name} | "
            f"trained={bundle.get('metadata', {}).get('training_date', 'unknown')}"
        )
        return bundle

    # ------------------------------------------------------------------
    # List and management
    # ------------------------------------------------------------------

    def list_models(
        self,
        symbol: str | None = None,
        model_type: str | None = None,
    ) -> list[dict]:
        """Return metadata for all saved models, optionally filtered."""
        results = []
        for path in sorted(self._root.glob("*.pkl")):
            parsed = self._parse_filename(path.name)
            if parsed is None:
                continue
            if symbol and parsed["symbol"] != symbol:
                continue
            if model_type and parsed["model_type"] != model_type:
                continue
            results.append({
                "path": str(path),
                "filename": path.name,
                **parsed,
            })
        return results

    def delete(self, symbol: str, timeframe: str, model_type: str, version: int) -> bool:
        path = self._root / self._filename(symbol, timeframe, model_type, version)
        if path.exists():
            path.unlink()
            logger.info(f"ModelRegistry deleted | {path.name}")
            return True
        return False

    def get_latest_version(
        self, symbol: str, timeframe: str, model_type: str
    ) -> int | None:
        path = self._latest_path(symbol, timeframe, model_type)
        if path is None:
            return None
        parsed = self._parse_filename(path.name)
        return parsed["version"] if parsed else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _filename(symbol: str, timeframe: str, model_type: str, version: int) -> str:
        return f"{symbol}_{timeframe}_{model_type}_v{version:03d}.pkl"

    def _next_version(self, symbol: str, timeframe: str, model_type: str) -> int:
        current = self.get_latest_version(symbol, timeframe, model_type)
        return (current or 0) + 1

    def _latest_path(
        self, symbol: str, timeframe: str, model_type: str
    ) -> Path | None:
        pattern = f"{symbol}_{timeframe}_{model_type}_v*.pkl"
        matches = sorted(self._root.glob(pattern))
        return matches[-1] if matches else None

    @staticmethod
    def _parse_filename(filename: str) -> dict | None:
        pattern = r"^(.+)_(.+)_(.+)_v(\d+)\.pkl$"
        m = re.match(pattern, filename)
        if not m:
            return None
        return {
            "symbol": m.group(1),
            "timeframe": m.group(2),
            "model_type": m.group(3),
            "version": int(m.group(4)),
        }