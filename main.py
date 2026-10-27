"""
main.py
-------
Live trading entry point — intraday branch.

Modes (set TRADING_MODE in .env):
  paper    : live market data, simulated fills (default, safe)
  demo     : live market data, real fills via MT5 DEMO account
  live     : live market data, real fills via MT5 (real money)
  backtest : run a historical backtest and print metrics

Intraday configuration:
  - 16 strategy instances: 4 symbols × 2 strategies × 2 timeframes (M15, H1)
  - Strategy state seeded from historical bars on startup (no cold-start WAITING)
  - Hot-reload of config/live_params.json (edit params without restart)
  - Auto-optimizer triggers every 20 closed trades
  - Signal diagnostics written to logs/signal_diagnostics.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from strategies.ml_strategy import MLStrategy
from risk.engine import RiskEngine, AccountState, PositionInfo
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
# Trading system
# ---------------------------------------------------------------------------

class TradingSystem:
    """
    Orchestrates the full live/paper trading pipeline.

    Fixes applied in this version:
      1. Strategy state seeded from historical bars on startup
         → trend strategy never shows "WAITING" after first bar
      2. Strategy warmup_bars counter advanced from history
         → strategies fire from bar 1, not bar 50+
      3. M15 included in signal diagnostics timeframe filter
      4. ADX ceiling raised in live_params.json (external)
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
            config=PipelineConfig(compute_ml=True),
            feature_store=self._feature_store,
        )

        # Signal diagnostics (FIX 3: include M15 so M15 signals are visible)
        from monitoring.signal_diagnostics import SignalDiagnostics
        self._diag = SignalDiagnostics(
            log_dir=Path("logs"),
            log_timeframes=["M15", "H1"],   # was ["H1","H4"] — now includes M15
        )

        # Load hot-reloadable strategy params from config/live_params.json
        self._params_file = Path(__file__).parent / "config" / "live_params.json"
        self._params = self._load_live_params()
        try:
            self._last_params_mtime: float = self._params_file.stat().st_mtime
        except Exception:
            self._last_params_mtime: float = 0.0

        # 16 strategy instances: 4 symbols × 2 strategies × 2 timeframes
        self._strategies = self._build_strategies()

        # Learning loop state — persisted to disk so restarts don't re-trigger
        self._closed_trade_count: int = 0
        self._optimize_every_n_trades: int = 20
        self._optimizer_state_file = Path(__file__).parent / "logs" / "optimizer_state.json"
        self._last_optimization_count: int = self._load_optimizer_state()

        # Risk engine
        rp = self._params.get("risk", {})
        self._risk = RiskEngine(
            max_risk_per_trade=rp.get("max_risk_per_trade",
                                      settings.risk.max_risk_per_trade),
            max_total_exposure=settings.risk.max_total_exposure,
            max_daily_drawdown=settings.risk.max_daily_drawdown,
            max_peak_drawdown=settings.risk.max_peak_drawdown,
            min_edge_cost_ratio=rp.get("min_edge_cost_ratio",
                                       settings.risk.min_edge_cost_ratio),
            max_positions=rp.get("max_positions", 8),
        )

        # Broker
        if mode in ("demo", "live"):
            self._broker = MT5Broker(
                login=settings.mt5.login,
                password=settings.mt5.password,
                server=settings.mt5.server,
                max_spread_points=rp.get("max_spread_points", 40.0),
                max_spread_pct=rp.get("max_spread_pct", 0.0010),
            )
        else:
            self._broker = PaperBroker(
                initial_equity=10_000.0,
                commission_per_lot=7.0,
            )

        # Reconciler — only used in demo/live mode
        self._reconciler = None
        if mode in ("demo", "live"):
            from execution.reconciler import PositionReconciler
            self._reconciler = PositionReconciler(self._broker)

        # Last known account state
        self._account_state: AccountState | None = None

    # ------------------------------------------------------------------
    # Main async trading loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        setup_logging()

        logger.info(
            f"TradingSystem starting | mode={self._mode} | "
            f"strategies=[{len(self._strategies)} instances]"
        )

        # Connect broker
        self._broker.connect()

        # Startup reconciliation
        if self._reconciler is not None:
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

        # Initialise drawdown breaker — use persisted peak to survive restarts
        peak_equity = self._load_peak_equity(equity)
        self._risk.drawdown_breaker.initialize(equity=peak_equity)
        logger.info(f"DrawdownBreaker peak baseline: ${peak_equity:,.2f} "
                    f"(current equity: ${equity:,.2f})")

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

            # FIX 1 + 2: Seed strategy state from historical features
            logger.info("Seeding strategy state from historical data...")
            self._seed_strategy_state()

            # Stream new bars
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
        logger.warning(
            f"MT5Feed reconnected (attempt #{attempt}) — "
            "MT5Feed._last_seen deduplication will skip already-seen bars"
        )

    # ------------------------------------------------------------------
    # FIX 1 + 2: Seed strategy state from historical bars
    # ------------------------------------------------------------------

    def _seed_strategy_state(self) -> None:
        """
        Feed historical feature rows to each strategy so that:
          1. _bar_count is advanced past warmup_bars threshold
          2. _prev_fast / _prev_slow are populated (no WAITING on bar 1)

        Without this, trend strategies show "WAITING: no previous bar data yet"
        on every bar after restart because they need bar T-1 to compare with bar T.
        """
        seeded = 0
        for strategy in self._strategies:
            sym = strategy.symbol
            tf  = strategy.timeframe

            features_df = self._pipeline.compute(sym, tf)
            if features_df is None or len(features_df) == 0:
                continue

            # Feed the last min(warmup_bars+2, len) rows to advance state
            # We use warmup_bars + 2 to ensure crossover state is populated
            warmup = getattr(strategy, "warmup_bars", 50)
            n_seed = min(warmup + 2, len(features_df))
            seed_rows = features_df.tail(n_seed).to_dicts()

            for row in seed_rows:
                # Call on_bar_update but don't submit any orders from history
                _ = strategy.on_bar_update(row)

            seeded += 1

        # Reset all internal position state after seeding
        # Seeding may leave strategies believing they hold positions
        # (from historical crossovers/entries) — clear that before live trading
        for strategy in self._strategies:
            if hasattr(strategy, "_current_position"):
                strategy._current_position = None
            if hasattr(strategy, "current_position"):
                strategy.current_position = None
            if hasattr(strategy, "_bar_count"):
                # Keep bar_count (warmup passed) but clear position state
                pass

        logger.info(
            f"Strategy seeding complete | {seeded}/{len(self._strategies)} "
            f"strategies seeded from historical data | position state cleared"
        )

    # ------------------------------------------------------------------
    # Per-bar processing
    # ------------------------------------------------------------------

    async def _on_bar(self, bar) -> None:
        """Process one bar through the full pipeline."""
        # 1. Store raw bar
        self._raw_store.write(bar)

        # 2. Update broker price (paper mode P&L tracking)
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

        # 3b. Hot-reload strategies if live_params.json changed
        self._reload_strategies_if_needed()

        # 4. Update equity via reconciler (demo/live) or broker info (paper)
        if self._reconciler is not None:
            self._account_state, _ = self._reconciler.reconcile(
                system_positions=(
                    self._account_state.open_positions
                    if self._account_state else []
                )
            )
            account_state = self._account_state
        else:
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

        # Learning loop: trigger optimizer every N closed trades
        self._check_and_optimize(account_state)

        # 5. Run strategies
        for strategy in self._strategies:
            if strategy.symbol != bar.symbol or strategy.timeframe != bar.timeframe:
                continue

            signal = strategy.on_bar_update(row)

            # Log what the strategy saw (FIX 3: M15 now included in filter)
            strategy_type = (
                "trend" if "trend" in strategy.strategy_id
                else "mean_reversion" if "mean_rev" in strategy.strategy_id
                else "ml"
            )
            extra_state: dict = {}
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
                logger.info(
                    f"Signal REJECTED | {signal.strategy_id} | "
                    f"{signal.symbol}/{signal.timeframe} | "
                    f"{signal.direction.value} | "
                    f"reason={result.rejection_reason}"
                )
                continue

            # Submit order
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

    # ------------------------------------------------------------------
    # Live param helpers
    # ------------------------------------------------------------------

    def _load_live_params(self) -> dict:
        try:
            return json.loads(self._params_file.read_text())
        except Exception as e:
            logger.warning(f"Could not load live_params.json: {e} — using defaults")
            return {}

    def _check_param_reload(self) -> bool:
        try:
            mtime = self._params_file.stat().st_mtime
            if mtime > self._last_params_mtime:
                self._last_params_mtime = mtime
                return True
        except Exception:
            pass
        return False

    def _reload_strategies_if_needed(self) -> None:
        if self._check_param_reload():
            self._params = self._load_live_params()
            self._strategies = self._build_strategies()
            logger.info(
                f"Strategies reloaded from live_params.json "
                f"(v{self._params.get('_version', '?')})"
            )

    def _build_strategies(self) -> list:
        """
        Build 16 strategy instances:
        4 symbols × 2 strategies × 2 timeframes (M15, H1)
        """
        tp = self._params.get("trend", {})
        mr = self._params.get("mean_reversion", {})
        mlp = self._params.get("ml", {})
        enable_ml = mlp.get("enabled", True)

        symbols    = settings.mt5.symbols
        timeframes = ["M15", "H1"]

        strategies = []
        for sym in symbols:
            for tf in timeframes:
                strategies.append(TrendStrategy(
                    symbol=sym, timeframe=tf,
                    fast_period      =tp.get("fast_period", 9),
                    slow_period      =tp.get("slow_period", 21),
                    adx_threshold    =tp.get("adx_threshold", 18.0),
                    exit_adx_threshold=tp.get("exit_adx_threshold", 12.0),
                    atr_multiplier   =tp.get("atr_multiplier", 1.5),
                    tp_multiplier    =tp.get("tp_multiplier", 2.5),
                    cooldown_bars    =tp.get("cooldown_bars", 1),
                ))
                strategies.append(MeanReversionStrategy(
                    symbol=sym, timeframe=tf,
                    oversold_rsi    =mr.get("oversold_rsi", 38.0),
                    overbought_rsi  =mr.get("overbought_rsi", 62.0),
                    adx_ceiling     =mr.get("adx_ceiling", 45.0),
                    atr_multiplier  =mr.get("atr_multiplier", 1.5),
                    max_bars_in_trade=mr.get("max_bars_in_trade", 16),
                    min_band_width  =mr.get("min_band_width", 0.0005),
                ))
                if enable_ml:
                    strategies.append(MLStrategy(
                        symbol=sym, timeframe=tf,
                        confidence_threshold=mlp.get("confidence_threshold", 0.40),
                        atr_multiplier      =mlp.get("atr_multiplier", 1.5),
                        tp_multiplier       =mlp.get("tp_multiplier", 2.5),
                        cooldown_bars       =mlp.get("cooldown_bars", 1),
                ))

        logger.info(
            f"Built {len(strategies)} strategy instances | "
            f"symbols={symbols} | timeframes={timeframes}"
        )
        return strategies

    def stop(self) -> None:
        self._running = False

    def _load_peak_equity(self, current_equity: float) -> float:
        """
        Load the persisted peak equity so drawdown is measured from the
        true historical peak, not just the current session start.
        This prevents the circuit breaker from being reset on every restart.
        """
        peak_file = Path(__file__).parent / "logs" / "peak_equity.json"
        try:
            import json
            peak_file.parent.mkdir(exist_ok=True)
            if peak_file.exists():
                data = json.loads(peak_file.read_text())
                stored_peak = float(data.get("peak_equity", current_equity))
                # If current equity exceeds stored peak, update it
                new_peak = max(stored_peak, current_equity)
            else:
                new_peak = current_equity
            # Always save the latest peak
            peak_file.write_text(json.dumps({
                "peak_equity": new_peak,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }))
            return new_peak
        except Exception as e:
            logger.warning(f"Could not load peak equity: {e} — using current")
            return current_equity

    def _load_optimizer_state(self) -> int:
        """Load last optimization trade count from disk."""
        try:
            import json
            self._optimizer_state_file.parent.mkdir(exist_ok=True)
            if self._optimizer_state_file.exists():
                data = json.loads(self._optimizer_state_file.read_text())
                count = int(data.get("last_optimization_count", 0))
                logger.debug(f"LearningLoop: loaded last_optimization_count={count}")
                return count
        except Exception:
            pass
        return 0

    def _save_optimizer_state(self, count: int) -> None:
        """Persist optimization state to survive restarts."""
        try:
            import json
            self._optimizer_state_file.parent.mkdir(exist_ok=True)
            self._optimizer_state_file.write_text(
                json.dumps({"last_optimization_count": count,
                            "saved_at": datetime.now(tz=timezone.utc).isoformat()})
            )
        except Exception as e:
            logger.debug(f"LearningLoop: failed to save state: {e}")

    # ------------------------------------------------------------------
    # Learning loop
    # ------------------------------------------------------------------

    def _check_and_optimize(self, account_state) -> None:
        try:
            import MetaTrader5 as mt5
            if mt5.terminal_info() is None:
                return
            from datetime import timedelta
            now   = datetime.now(tz=timezone.utc)
            start = now - timedelta(days=30)
            raw   = mt5.history_deals_get(start, now)
            if raw is None:
                return
            entry_pos = set()
            exit_pos  = set()
            for d in raw:
                if d.entry == mt5.DEAL_ENTRY_IN:
                    entry_pos.add(d.position_id)
                elif d.entry == mt5.DEAL_ENTRY_OUT:
                    exit_pos.add(d.position_id)
            closed_count = len(entry_pos & exit_pos)
            self._closed_trade_count = closed_count

            new_trades = closed_count - self._last_optimization_count
            if new_trades >= self._optimize_every_n_trades:
                logger.info(
                    f"LearningLoop: {new_trades} new trades since last run "
                    f"— triggering auto_optimizer"
                )
                try:
                    from scripts.auto_optimizer import run_optimization
                    updated = run_optimization()
                    self._last_optimization_count = closed_count
                    self._save_optimizer_state(closed_count)
                    if updated:
                        self._reload_strategies_if_needed()
                except Exception as e:
                    logger.warning(f"LearningLoop: optimizer error: {e}")
        except Exception as e:
            logger.debug(f"LearningLoop: trade count check failed: {e}")


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(args: argparse.Namespace) -> None:
    from backtesting.engine import BacktestEngine, BacktestConfig
    from features.pipeline import PipelineConfig

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

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
        symbol=args.symbol, timeframe=args.timeframe,
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
    parser = argparse.ArgumentParser(description="TradingBotMT5 — intraday")
    parser.add_argument(
        "--mode", choices=["paper", "demo", "live", "backtest"],
        default=settings.mode,
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--equity", type=float, default=10_000.0)
    return parser.parse_args()


async def _run_with_shutdown(mode: str) -> None:
    """
    Run the trading system with clean asyncio shutdown.
    Handles Ctrl+C gracefully without leaving pending tasks.
    """
    system = TradingSystem(mode=mode)
    loop   = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.info(f"Shutdown signal received ({sig})")
        system.stop()
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run system until stopped
    task = asyncio.create_task(system.run())
    await asyncio.wait([task, asyncio.create_task(stop_event.wait())],
                       return_when=asyncio.FIRST_COMPLETED)

    # Cancel any remaining tasks cleanly
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    logger.info("TradingSystem shutdown complete")


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.mode == "backtest":
        run_backtest(args)
        return

    try:
        asyncio.run(_run_with_shutdown(mode=args.mode))
    except KeyboardInterrupt:
        pass  # Already handled by signal handler


if __name__ == "__main__":
    main()