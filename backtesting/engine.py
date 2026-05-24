"""
backtesting/engine.py
----------------------
Event-driven backtesting engine.

Wires together: DataStore → FeaturePipeline → Strategy → RiskEngine
               → FillModel → Metrics

Key design principles:
  1. Strict time ordering: bar T features only use bars 0..T
  2. Signal-to-fill delay: signals from bar T enter at bar T+1 open
  3. SL/TP checked before new signals on each bar (bar processing order)
  4. Identical code path to live trading (only the broker is swapped)
  5. No global state: each BacktestEngine instance is self-contained

Event loop per bar (strict order):
  ┌─────────────────────────────────────────────────────────┐
  │  Bar T received                                         │
  │  1. Update paper broker price (bar.close)               │
  │  2. Check all open positions for SL/TP hit              │
  │     → if hit: close position, record trade              │
  │  3. If pending signal from bar T-1:                     │
  │     → fill at bar T open price                          │
  │     → open position                                     │
  │  4. Compute features up to bar T                        │
  │  5. Run strategy.on_bar(features[-1])                   │
  │     → if signal: pass to risk engine                    │
  │     → if approved: store as pending (fills next bar)    │
  │  6. Record equity snapshot                              │
  └─────────────────────────────────────────────────────────┘

Point-in-time guarantee:
  Features at step 4 are computed on bars 0..T.
  The strategy cannot see bar T+1 data.
  The fill at step 3 uses bar T's open — not bar T's close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl
from loguru import logger

from feeds.base_feed import Bar
from features.pipeline import FeaturePipeline, PipelineConfig
from strategies.base_strategy import BaseStrategy, Direction, Signal
from risk.engine import RiskEngine, AccountState, PositionInfo
from risk.order import Order, OrderStatus
from backtesting.fill_model import FillModel, OpenPosition, FillType, BarFill
from backtesting.metrics import BacktestMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_equity: float = 10_000.0
    commission_per_lot: float = 7.0
    spread_points: float = 2.0
    bars_per_year: int = 252 * 24    # hourly bars; adjust per timeframe
    warmup_bars: int = 250           # bars used for indicator warm-up only
    verbose: bool = False            # log every bar

    @staticmethod
    def for_timeframe(timeframe: str) -> int:
        """Return bars_per_year for common timeframes."""
        mapping = {
            "M1": 252 * 24 * 60,
            "M5": 252 * 24 * 12,
            "M15": 252 * 24 * 4,
            "H1": 252 * 24,
            "H4": 252 * 6,
            "D1": 252,
        }
        return mapping.get(timeframe, 252 * 24)


# ---------------------------------------------------------------------------
# Internal trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    entry_bar: int
    exit_bar: int
    stop_loss: float | None
    take_profit: float | None
    strategy_id: str
    fill_type: str             # "sl", "tp", "signal", "gap", "end_of_test"
    gross_pnl: float
    commission: float
    net_pnl: float

    @property
    def bars_held(self) -> int:
        return self.exit_bar - self.entry_bar


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Usage:
        config = BacktestConfig(
            symbol="XAUUSD", timeframe="H1",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            initial_equity=10_000,
        )
        engine = BacktestEngine(
            config=config,
            strategy=TrendStrategy("XAUUSD", "H1"),
            raw_store=store,
        )
        metrics = engine.run()
        print(metrics)
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: BaseStrategy,
        raw_store,                       # RawDataStore
        risk_config: dict | None = None,
        pipeline_config: PipelineConfig | None = None,
        fill_model: FillModel | None = None,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._store = raw_store
        self._fill_model = fill_model or FillModel(
            slippage_factor=0.1,
            gap_fill=True,
            spread_points={config.symbol: config.spread_points},
        )

        rc = risk_config or {}
        self._risk = RiskEngine(
            commission_per_lot=config.commission_per_lot,
            **rc,
        )

        self._pipeline = FeaturePipeline(
            store=raw_store,
            config=pipeline_config or PipelineConfig(compute_ml=False),
        )

        # Internal state
        self._equity = config.initial_equity
        self._balance = config.initial_equity
        self._open_positions: dict[str, OpenPosition] = {}
        self._pending_order: Order | None = None   # fills at next bar open
        self._trade_log: list[TradeRecord] = []
        self._equity_curve: list[float] = []
        self._bar_index: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> BacktestMetrics:
        """
        Run the full backtest. Returns computed metrics.
        """
        self._strategy.reset()
        self._risk.drawdown_breaker.initialize(equity=self._config.initial_equity)

        logger.info(
            f"Backtest starting | {self._config.symbol}/{self._config.timeframe} | "
            f"{self._config.start_date.date()} → {self._config.end_date.date()} | "
            f"equity={self._config.initial_equity:,.0f}"
        )

        # Load all bars for the period (including warmup)
        all_bars = self._load_bars()
        if len(all_bars) < self._config.warmup_bars + 2:
            logger.error(
                f"Insufficient bars: {len(all_bars)} < "
                f"{self._config.warmup_bars + 2} required"
            )
            return BacktestMetrics()

        # Split: warmup bars are used to seed features but not traded
        warmup = all_bars[:self._config.warmup_bars]
        trading_bars = all_bars[self._config.warmup_bars:]

        logger.info(
            f"Loaded {len(all_bars)} bars | "
            f"warmup={len(warmup)} | trading={len(trading_bars)}"
        )

        # ── Event loop ──────────────────────────────────────────────
        bar_window = list(warmup)  # rolling window fed to features

        for bar in trading_bars:
            bar_window.append(bar)
            self._bar_index += 1
            self._process_bar(bar, bar_window)

        # Close all open positions at end of test
        self._close_all_at_end(trading_bars[-1] if trading_bars else None)

        # ── Compute and return metrics ───────────────────────────────
        trade_dicts = [self._trade_to_dict(t) for t in self._trade_log]
        metrics = compute_metrics(
            equity_curve=self._equity_curve,
            trade_log=trade_dicts,
            bars_per_year=self._config.bars_per_year,
        )

        logger.info(
            f"Backtest complete | trades={metrics.total_trades} | "
            f"return={metrics.total_return:.2%} | sharpe={metrics.sharpe:.2f} | "
            f"maxdd={metrics.max_drawdown:.2%}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Per-bar processing
    # ------------------------------------------------------------------

    def _process_bar(self, bar: Bar, bar_window: list[Bar]) -> None:
        """Process one bar through the full event pipeline."""

        # Step 1: Check open positions for SL/TP
        positions_to_close = []
        for sym, pos in list(self._open_positions.items()):
            if sym != bar.symbol:
                continue
            result = self._fill_model.check_position(bar, pos)
            if result.triggered:
                positions_to_close.append((pos, result))

        for pos, result in positions_to_close:
            self._close_position(pos, result, self._bar_index)

        # Step 2: Fill pending order at this bar's open
        if self._pending_order is not None:
            self._fill_pending(bar, self._bar_index)
            self._pending_order = None

        # Step 3: Compute features and run strategy
        features_df = self._compute_features(bar_window)
        if features_df is not None and len(features_df) > 0:
            row = features_df.tail(1).to_dicts()[0]
            row["close"] = bar.close  # ensure current close in meta

            # Update drawdown breaker
            self._risk.drawdown_breaker.update(equity=self._equity)

            # Build account state
            account = self._build_account_state()

            # Get signal from strategy
            signal = self._strategy.on_bar_update(row)
            if signal is not None:
                result = self._risk.evaluate(signal, account)
                if result.approved and result.order is not None:
                    if signal.is_exit:
                        # Exit: close immediately at this bar's close
                        self._handle_exit_signal(signal, bar)
                    else:
                        # Entry: store as pending, fills next bar open
                        self._pending_order = result.order

        # Step 4: Record equity snapshot
        self._equity_curve.append(self._calculate_equity(bar))

        if self._config.verbose:
            logger.debug(
                f"Bar {self._bar_index} | {bar.event_time} | "
                f"close={bar.close:.4f} | equity={self._equity:.2f} | "
                f"positions={len(self._open_positions)}"
            )

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, bar_window: list[Bar]) -> pl.DataFrame | None:
        """
        Compute features on the current bar window.
        We build a temporary DataFrame from bars rather than hitting the
        disk store every bar — much faster for backtesting.
        """
        if len(bar_window) < self._pipeline.config.min_bars:
            return None

        rows = [b.to_dict() for b in bar_window]
        df = pl.DataFrame(rows)

        try:
            from features.technical import add_all_technical
            from features.ml_features import add_all_ml_features

            cfg = self._pipeline.config
            if cfg.compute_technical:
                df = add_all_technical(
                    df,
                    ema_periods=cfg.ema_periods,
                    sma_periods=cfg.sma_periods,
                    rsi_period=cfg.rsi_period,
                    atr_period=cfg.atr_period,
                    bb_period=cfg.bb_period,
                    macd_params=cfg.macd_params,
                    stoch_params=cfg.stoch_params,
                    adx_period=cfg.adx_period,
                    include_vwap=cfg.include_vwap,
                )

            # Drop warmup NaN rows — only return the last valid row
            required = [c for c in cfg.required_feature_cols if c in df.columns]
            if required:
                df = df.drop_nulls(subset=required)

        except Exception as e:
            logger.warning(f"Feature computation error: {e}")
            return None

        return df if len(df) > 0 else None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _fill_pending(self, bar: Bar, bar_index: int) -> None:
        """Fill a pending order at this bar's open price."""
        order = self._pending_order
        if order is None:
            return

        fill_price = self._fill_model.entry_fill_price(bar, order.direction)
        commission = self._config.commission_per_lot * order.lot_size
        self._balance -= commission

        pos = OpenPosition(
            symbol=order.symbol,
            direction=order.direction,
            entry_price=fill_price,
            lot_size=order.lot_size,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            entry_bar_index=bar_index,
            strategy_id=order.strategy_id,
        )
        self._open_positions[order.symbol] = pos

        logger.debug(
            f"Fill | {order.symbol} {order.direction.value} | "
            f"price={fill_price:.4f} lots={order.lot_size:.2f} "
            f"SL={order.stop_loss} TP={order.take_profit}"
        )

    def _close_position(
        self,
        pos: OpenPosition,
        fill_result: BarFill,
        bar_index: int,
    ) -> None:
        """Close a position, record the trade, update balance."""
        fill_price = fill_result.fill_price or pos.entry_price
        gross_pnl = self._calc_pnl(pos, fill_price)
        commission = self._config.commission_per_lot * pos.lot_size
        net_pnl = gross_pnl - commission

        self._balance += net_pnl
        del self._open_positions[pos.symbol]

        fill_type_str = fill_result.fill_type.value if fill_result.fill_type else "unknown"
        record = TradeRecord(
            symbol=pos.symbol,
            direction=pos.direction.value,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            lot_size=pos.lot_size,
            entry_bar=pos.entry_bar_index,
            exit_bar=bar_index,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            strategy_id=pos.strategy_id,
            fill_type=fill_type_str,
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
        )
        self._trade_log.append(record)

        logger.debug(
            f"Close | {pos.symbol} {fill_type_str} | "
            f"entry={pos.entry_price:.4f} exit={fill_price:.4f} "
            f"net_pnl={net_pnl:.2f}"
        )

    def _handle_exit_signal(self, signal: Signal, bar: Bar) -> None:
        """Handle strategy exit signals — close position at bar close."""
        pos = self._open_positions.get(signal.symbol)
        if pos is None:
            return
        fill = BarFill(
            triggered=True,
            fill_type=FillType.SIGNAL,
            fill_price=bar.close,
        )
        self._close_position(pos, fill, self._bar_index)

    def _close_all_at_end(self, last_bar: Bar | None) -> None:
        """Force-close all positions at end of backtest."""
        for sym, pos in list(self._open_positions.items()):
            price = last_bar.close if last_bar else pos.entry_price
            fill = BarFill(
                triggered=True,
                fill_type=FillType.SIGNAL,
                fill_price=price,
            )
            self._close_position(pos, fill, self._bar_index)
            logger.debug(f"End-of-test close: {sym} at {price:.4f}")

    # ------------------------------------------------------------------
    # Account state helpers
    # ------------------------------------------------------------------

    def _build_account_state(self) -> AccountState:
        positions = [
            PositionInfo(
                symbol=pos.symbol,
                direction=pos.direction,
                lot_size=pos.lot_size,
                entry_price=pos.entry_price,
                strategy_id=pos.strategy_id,
                open_time=datetime.now(tz=timezone.utc),
            )
            for pos in self._open_positions.values()
        ]
        return AccountState(
            equity=self._equity,
            balance=self._balance,
            open_positions=positions,
        )

    def _calculate_equity(self, bar: Bar) -> float:
        """Equity = balance + unrealized P&L from open positions."""
        unrealized = sum(
            self._calc_pnl(pos, bar.close)
            for pos in self._open_positions.values()
            if pos.symbol == bar.symbol
        )
        self._equity = self._balance + unrealized
        return self._equity

    def _calc_pnl(self, pos: OpenPosition, price: float) -> float:
        """Calculate P&L for a position at a given price."""
        pip_values = {
            "XAUUSD": 100.0, "EURUSD": 10.0,
            "GBPUSD": 10.0, "BTCUSDT": 1.0,
        }
        pip_value = pip_values.get(pos.symbol, 10.0)
        if pos.direction == Direction.LONG:
            return (price - pos.entry_price) * pip_value * pos.lot_size
        else:
            return (pos.entry_price - price) * pip_value * pos.lot_size

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_bars(self) -> list[Bar]:
        """Load bars from the raw store for the backtest period."""
        df = self._store.read(
            self._config.symbol,
            self._config.timeframe,
            start=self._config.start_date,
            end=self._config.end_date,
            as_polars=True,
        )
        if df is None or len(df) == 0:
            return []

        bars = []
        for row in df.iter_rows(named=True):
            bars.append(Bar(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                event_time=row["event_time"],
                publish_time=row["publish_time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                spread=row.get("spread", 0.0),
                source=row.get("source", ""),
            ))
        return bars

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_to_dict(t: TradeRecord) -> dict:
        return {
            "direction": t.direction,
            "net_pnl": t.net_pnl,
            "commission": t.commission,
            "bars_held": t.bars_held,
        }

    @property
    def trade_log(self) -> list[TradeRecord]:
        return list(self._trade_log)

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)