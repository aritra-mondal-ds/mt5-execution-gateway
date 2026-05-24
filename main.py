"""
main.py
-------
Live trading entry point.

Wires together all layers:
  MT5Feed → FeaturePipeline → Strategies → RiskEngine → MT5Broker/PaperBroker

Modes (set TRADING_MODE in .env):
  paper    : live market data, simulated fills (default, safe)
  demo     : live market data, real fills via MT5 DEMO account
  live     : live market data, real fills via MT5 (real money)
  backtest : run a historical backtest and print metrics

Usage:
  conda activate tradingbotmt5
  python main.py               # paper trading (default)
  python main.py --mode demo   # demo trading — REAL MT5 fills, demo money
  python main.py --mode live   # live trading — REAL money
  python main.py --mode backtest --start 2023-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config.settings import settings
from data.store import RawDataStore
from data.feature_store import FeatureStore
from feeds.mt5_feed import MT5Feed
from features.pipeline import FeaturePipeline, PipelineConfig
from strategies.trend import TrendStrategy
from strategies.mean_reversion import MeanReversionStrategy
from risk.engine import RiskEngine
from execution.paper_broker import PaperBroker
from execution.mt5_broker import MT5Broker


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=settings.log.rotation,
        retention=settings.log.retention,
        compression="zip",
    )


# ---------------------------------------------------------------------------
# Live / paper trading loop
# ---------------------------------------------------------------------------

class TradingSystem:
    """
    Orchestrates the full live/paper trading pipeline.

    Bar processing per symbol per timeframe:
      1. New bar arrives from MT5Feed
      2. Store raw bar to RawDataStore
      3. Compute features via FeaturePipeline
      4. Run each active strategy on latest feature row
      5. Pass signals to RiskEngine
      6. Submit approved orders to broker
      7. Update drawdown breaker with current equity
    """

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode
        self._running = False

        # Data layer
        self._raw_store = RawDataStore(root=settings.data.raw_dir)
        self._feature_store = FeatureStore(root=settings.data.features_dir)

        # Feature pipeline
        self._pipeline = FeaturePipeline(
            store=self._raw_store,
            config=PipelineConfig(compute_ml=False),
            feature_store=self._feature_store,
        )

        # Strategies (add/remove strategies here)
        self._strategies = [
            TrendStrategy(
                symbol="XAUUSD", timeframe="H1",
                fast_period=9, slow_period=21,
                adx_threshold=25.0,
            ),
            MeanReversionStrategy(
                symbol="XAUUSD", timeframe="H1",
                oversold_rsi=35.0, overbought_rsi=65.0,
            ),
        ]

        # Risk engine
        self._risk = RiskEngine(
            max_risk_per_trade=settings.risk.max_risk_per_trade,
            max_total_exposure=settings.risk.max_total_exposure,
            max_daily_drawdown=settings.risk.max_daily_drawdown,
            max_peak_drawdown=settings.risk.max_peak_drawdown,
            min_edge_cost_ratio=settings.risk.min_edge_cost_ratio,
        )

        # Broker — demo and live both use MT5Broker (real fills)
        # paper uses PaperBroker (simulated fills, no MT5 orders submitted)
        if mode in ("demo", "live"):
            self._broker = MT5Broker(
                login=settings.mt5.login,
                password=settings.mt5.password,
                server=settings.mt5.server,
            )
        else:
            self._broker = PaperBroker(
                initial_equity=10_000.0,
                commission_per_lot=7.0,
            )

        # Signal diagnostics
        from monitoring.signal_diagnostics import SignalDiagnostics
        self._diag = SignalDiagnostics(
            log_dir=Path("logs"),
            log_timeframes=["H1", "H4"],   # only log higher timeframes
        )

        # Reconciler — only used in demo/live mode
        self._reconciler = None
        if mode in ("demo", "live"):
            from execution.reconciler import PositionReconciler
            self._reconciler = PositionReconciler(self._broker)

        # Last known account state — updated by reconciler each bar
        self._account_state: "AccountState | None" = None

    async def run(self) -> None:
        """Main async trading loop."""
        self._running = True
        setup_logging()

        logger.info(
            f"TradingSystem starting | mode={self._mode} | "
            f"strategies={[s.strategy_id for s in self._strategies]}"
        )

        # Connect broker
        self._broker.connect()

        # Startup reconciliation — reads all open MT5 positions before trading
        if self._reconciler is not None:
            from risk.engine import AccountState
            logger.info("Reconciling open positions from MT5...")
            self._account_state, report = self._reconciler.reconcile(
                system_positions=[]
            )
            equity = self._account_state.equity
            logger.info(
                f"Reconciliation complete | "
                f"open_positions={len(self._account_state.open_positions)} | "
                f"equity={equity:.2f}"
            )
            for pos in self._account_state.open_positions:
                logger.info(
                    f"  Existing position: {pos.symbol} "
                    f"{pos.direction.value} {pos.lot_size:.2f} lots "
                    f"@ {pos.entry_price:.4f} [{pos.strategy_id}]"
                )
        else:
            account = self._broker.get_account_info()
            equity = account.equity

        # Initialise drawdown breaker with real equity
        self._risk.drawdown_breaker.initialize(equity=equity)

        # Wrap MT5Feed with automatic reconnection
        from feeds.reconnect import ReconnectingFeed
        from monitoring.alerts import AlertManager
        _alerts = AlertManager()

        base_feed = MT5Feed(
            login=settings.mt5.login,
            password=settings.mt5.password,
            server=settings.mt5.server,
        )

        feed = ReconnectingFeed(
            feed=base_feed,
            max_attempts=10,
            base_delay=2.0,
            max_delay=120.0,
            max_stale_seconds=180.0,
            health_check_interval=30.0,
            on_reconnect=self._on_reconnect_callback,
            on_circuit_open=lambda: _alerts.connection_lost(
                "MT5Feed", "Circuit breaker opened after 10 failures"
            ),
        )

        async with feed:
            # Warm up: load historical bars for each symbol/timeframe
            for symbol in settings.mt5.symbols:
                for tf in settings.mt5.timeframes:
                    logger.info(f"Warming up {symbol}/{tf}...")
                    bars = await feed.fetch_historical(
                        symbol, tf, settings.mt5.warmup_bars
                    )
                    self._raw_store.write_batch(bars)
                    logger.info(f"  {len(bars)} bars loaded")

            # Stream new bars with automatic reconnection on any error
            logger.info("Live streaming started. Press Ctrl+C to stop.")
            async for bar in feed.stream(
                symbols=settings.mt5.symbols,
                timeframes=settings.mt5.timeframes,
            ):
                if not self._running:
                    break
                await self._on_bar(bar)

        self._broker.disconnect()
        logger.info("TradingSystem stopped cleanly")

    async def _on_reconnect_callback(self, attempt: int) -> None:
        """Called by ReconnectingFeed after each successful reconnection."""
        logger.warning(
            f"MT5Feed reconnected (attempt #{attempt}) — "
            "MT5Feed._last_seen deduplication will skip already-seen bars"
        )

    async def _on_bar(self, bar) -> None:
        """Process one bar through the full pipeline."""
        # 1. Store raw bar
        self._raw_store.write(bar)

        # 2. Update broker price (for paper mode P&L tracking)
        if isinstance(self._broker, PaperBroker):
            self._broker.update_price_from_bar(
                bar.symbol, bar.close, bar.spread
            )

        # 3. Compute features
        features_df = self._pipeline.compute_latest(bar.symbol, bar.timeframe)
        if features_df is None or len(features_df) == 0:
            return

        row = features_df.to_dicts()[0]
        row["close"] = bar.close

        # 4. Update equity via reconciler (demo/live) or broker info (paper)
        if self._reconciler is not None:
            # Reconcile every bar — reads real MT5 positions and equity.
            # This catches manual trades, partial fills, and SL/TP hits
            # that happened outside the system.
            self._account_state, _ = self._reconciler.reconcile(
                system_positions=(
                    self._account_state.open_positions
                    if self._account_state else []
                )
            )
            account_state = self._account_state
        else:
            # Paper mode: build AccountState from PaperBroker ledger
            from risk.engine import AccountState, PositionInfo
            account_info = self._broker.get_account_info()
            positions = self._broker.get_positions()
            pos_infos = [
                PositionInfo(
                    symbol=p.symbol,
                    direction=p.direction,
                    lot_size=p.lot_size,
                    entry_price=p.open_price,
                    strategy_id="unknown",
                    open_time=p.open_time,
                )
                for p in positions
            ]
            account_state = AccountState(
                equity=account_info.equity,
                balance=account_info.balance,
                open_positions=pos_infos,
            )

        self._risk.drawdown_breaker.update(equity=account_state.equity)

        # 6. Run strategies and process signals
        for strategy in self._strategies:
            if strategy.symbol != bar.symbol or strategy.timeframe != bar.timeframe:
                continue

            signal = strategy.on_bar_update(row)

            # Log what the strategy saw regardless of signal outcome
            strategy_type = (
                "trend" if "trend" in strategy.strategy_id
                else "mean_reversion" if "mean_rev" in strategy.strategy_id
                else "ml"
            )
            extra_state = {}
            if hasattr(strategy, "_prev_fast"):
                extra_state = {
                    "fast_col": strategy._fast,
                    "slow_col": strategy._slow,
                    "adx_threshold": strategy.adx_threshold,
                    "prev_fast": strategy._prev_fast,
                    "prev_slow": strategy._prev_slow,
                }
            elif hasattr(strategy, "oversold_rsi"):
                extra_state = {
                    "adx_ceiling": strategy.adx_ceiling,
                    "oversold_rsi": strategy.oversold_rsi,
                    "overbought_rsi": strategy.overbought_rsi,
                }
            self._diag.log_evaluation(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy_type,
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                row=row,
                signal=signal,
                extra_state=extra_state,
            )

            if signal is None:
                continue

            result = self._risk.evaluate(signal, account_state)
            if not result.approved:
                logger.debug(
                    f"Signal rejected | {signal.strategy_id} | "
                    f"{result.rejection_reason}"
                )
                continue

            # 7. Submit order
            try:
                order = result.order
                ticket = self._broker.submit(order)
                logger.info(
                    f"Order submitted | {ticket} | "
                    f"{order.symbol} {order.direction.value} "
                    f"lots={order.lot_size:.2f}"
                )
            except Exception as e:
                logger.error(f"Order submission failed: {e}")

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(args: argparse.Namespace) -> None:
    """Run a historical backtest and print results."""
    from backtesting.engine import BacktestEngine, BacktestConfig
    from features.pipeline import PipelineConfig

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    raw_store = RawDataStore(root=settings.data.raw_dir)
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
        initial_equity=args.equity,
        warmup_bars=250,
        bars_per_year=BacktestConfig.for_timeframe(args.timeframe),
    )

    strategy = TrendStrategy(
        symbol=args.symbol,
        timeframe=args.timeframe,
        fast_period=9, slow_period=21,
    )

    engine = BacktestEngine(
        config=config,
        strategy=strategy,
        raw_store=raw_store,
        pipeline_config=PipelineConfig(compute_ml=False),
    )

    logger.info("Running backtest...")
    metrics = engine.run()
    print(metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TradingBotMT5")
    parser.add_argument(
        "--mode", choices=["paper", "demo", "live", "backtest"],
        default=settings.mode,
        help="Trading mode (default: from .env TRADING_MODE)",
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default="2023-01-01",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-01",
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Initial equity for backtest")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.mode == "backtest":
        run_backtest(args)
        return

    system = TradingSystem(mode=args.mode)

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()

    def handle_shutdown(sig, frame):
        logger.info(f"Shutdown signal received ({sig})")
        system.stop()
        loop.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        loop.run_until_complete(system.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()