"""
config/settings.py
------------------
Single source of truth for all configuration.
Loaded once at startup from .env / environment variables.
All other modules import from here — never read os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings          # pip install pydantic-settings

# ---------------------------------------------------------------------------
# Resolve project root and load .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # …/TradingBotMT5/
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Sub-models (grouped by concern)
# ---------------------------------------------------------------------------

class MT5Config(BaseModel):
    login: int
    password: str
    server: str
    timeout_ms: int = 10_000
    # Symbols this feed is responsible for
    symbols: list[str] = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    # Bar timeframes to stream/cache (MT5 TIMEFRAME constants as strings)
    timeframes: list[str] = ["M1", "M5", "M15", "H1", "H4", "D1"]
    # How many historical bars to load on startup
    warmup_bars: int = 500


class CryptoConfig(BaseModel):
    exchange: str = "binance"
    api_key: str = ""
    secret: str = ""
    symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    warmup_bars: int = 500
    use_testnet: bool = False


class DataConfig(BaseModel):
    # All paths relative to project root
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    signals_dir: Path = PROJECT_ROOT / "data" / "signals"
    models_dir: Path = PROJECT_ROOT / "models" / "registry"

    # Parquet compression — snappy is fast; zstd is smaller
    parquet_compression: Literal["snappy", "zstd", "gzip"] = "snappy"

    # Minimum bars required before a feature row is emitted
    min_bars_for_features: int = 200

    @field_validator("raw_dir", "features_dir", "signals_dir", "models_dir", mode="before")
    @classmethod
    def ensure_dir_exists(cls, v: Path | str) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


class RiskConfig(BaseModel):
    # Maximum fraction of account equity per trade
    max_risk_per_trade: float = float(os.environ.get("MAX_RISK_PER_TRADE", "0.01"))
    # Maximum total open exposure as fraction of equity
    max_total_exposure: float = float(os.environ.get("MAX_TOTAL_EXPOSURE", "0.10"))
    # Daily drawdown limit — trading halts if breached
    max_daily_drawdown: float = float(os.environ.get("MAX_DAILY_DRAWDOWN", "0.05"))
    # Peak-to-trough drawdown circuit breaker
    max_peak_drawdown: float = float(os.environ.get("MAX_PEAK_DRAWDOWN", "0.15"))
    # Minimum edge-to-cost ratio (expected return / expected cost)
    min_edge_cost_ratio: float = float(os.environ.get("MIN_EDGE_COST_RATIO", "1.5"))
    # Correlation threshold — reject trade if portfolio correlation > this
    max_portfolio_correlation: float = 0.70


class BacktestConfig(BaseModel):
    # Default spread in pips/points added to every fill
    default_spread_points: float = 2.0
    # Slippage as fraction of spread (0.5 = half a spread extra on market orders)
    slippage_factor: float = 0.5
    # Commission per lot (round-trip, in account currency)
    commission_per_lot: float = 7.0
    # Whether limit orders require price to trade *through* (not just touch)
    require_price_through: bool = True
    # Starting equity for backtests
    initial_equity: float = 10_000.0


class LogConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    rotation: str = "1 day"
    retention: str = "365 days"

    @field_validator("log_dir", mode="before")
    @classmethod
    def ensure_log_dir(cls, v: Path | str) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
# Root settings — reads from environment / .env
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Usage anywhere in the project:

        from config.settings import settings
        print(settings.mt5.server)
    """

    # Broker / exchange credentials come from .env
    mt5: MT5Config = Field(default_factory=lambda: MT5Config(
        login=int(os.environ.get("MT5_LOGIN", "0")),
        password=os.environ.get("MT5_PASSWORD", ""),
        server=os.environ.get("MT5_SERVER", ""),
    ))

    crypto: CryptoConfig = Field(default_factory=lambda: CryptoConfig(
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        secret=os.environ.get("BINANCE_SECRET", ""),
    ))

    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    # Live vs paper vs backtest mode
    mode: Literal["live", "demo", "paper", "backtest"] = Field(
        default=os.environ.get("TRADING_MODE", "paper")
    )

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        # Allow extra fields in .env without crashing
        extra = "ignore"


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()