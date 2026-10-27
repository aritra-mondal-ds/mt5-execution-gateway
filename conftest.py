"""
conftest.py
-----------
Global pytest configuration for TradingBotMT5.

Pre-loads C-extension libraries (LightGBM) to initialize Windows OpenMP
thread-local runtime descriptors before other native engines (Polars/Rayon/PyArrow)
allocate thread pools.
"""
from __future__ import annotations

try:
    import lightgbm as lgb
except ImportError:
    pass

