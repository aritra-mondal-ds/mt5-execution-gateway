# TradingBot MT5 – Full Project Documentation

This document is a standalone technical overview of the TradingBot MT5 project as inferred from the provided source files and logs. It is written for senior engineers and quants who may need to understand, operate, and extend the system.

---

## 1. System overview

TradingBot MT5 is an end‑to‑end automated trading stack that runs strategies on MetaTrader 5 accounts using a fully Pythonic research and execution environment.

At a high level the architecture comprises:

- **Data & storage layer**  
  - `RawDataStore`: time‑series OHLCV + spread + volume bars for each `(symbol, timeframe)` pair, powered by Polars.
  - `FeatureStore`: Parquet cache for derived feature DataFrames, partitioned by month and keyed by a configuration hash.

- **Feature engineering layer**  
  - `FeaturePipeline`: the single entry point for feature computation, orchestrating technical and ML feature functions with point‑in‑time correctness and data quality checks.

- **Strategy layer**  
  - `BaseStrategy` / `Signal` / `StrategyState`: clean abstractions that let strategies be stateful but side‑effect‑free functions from feature rows to `Signal` objects.
  - Concrete implementations, e.g. `MeanReversionStrategy`, implement specific trading logic.

- **Risk & order layer**  
  - Risk checks convert `Signal` → `Order` after sizing, cost filtering, and global risk constraints.
  - `CostFilter` models spread, commission, and slippage versus expected edge.
  - `Order` is the atomic unit passed to the execution layer.

- **Execution & broker integration**  
  - Execution components (not fully shown) translate `Order` into MT5 orders and positions, track fills, and reconcile account state.
  - `PerformanceTracker` reads MT5 closed deals and produces performance reports and a quantitative “ready for live” check.

- **Monitoring & diagnostics**  
  - `monitoring/dashboard.py`: Streamlit P&L and health dashboard that reads directly from the project’s data stores and log directories.
  - `SignalDiagnostics`: structured logging of every strategy evaluation with feature snapshots and human‑readable diagnoses.

The same FeaturePipeline + Strategy abstractions are used consistently across backtests, paper trading, and live trading to minimise sim‑to‑live drift.

---

## 2. Runtime modes and data flow

### 2.1 Modes

The system is designed to run in three primary modes (selected via configuration):

- **`paper`**: paper trading on a demo account or simulated broker, producing logs and diagnostics without real capital at risk.
- **`live`**: live trading on a real MT5 account, with strict readiness checks and risk limits.
- **`backtest`**: offline simulation using historical raw bars, reusing the exact same FeaturePipeline, strategies, and risk engine.

`settings.mode` is surfaced prominently in the Streamlit dashboard sidebar and in risk status panels so operators always know which mode they are monitoring.

### 2.2 Data flow

End‑to‑end, the data path is:

1. **Raw bars → features**  
   - The data acquisition layer (MT5 feed or historical loader) writes OHLCV bars into `RawDataStore`.
   - `FeaturePipeline.compute(...)` or `compute_latest(...)` loads raw bars, validates them, computes technical & ML features, filters NaNs, and optionally caches to `FeatureStore`.

2. **Features → signals**  
   - For each `(symbol, timeframe)` and evaluation time, a feature row is passed to each active `BaseStrategy` via `on_bar_update(row)`.
   - The strategy returns either a `Signal` or `None`.

3. **Signals → risk engine → orders**  
   - The risk engine consumes `Signal` objects, performs sizing, cost filtering, and global risk checks (drawdown, exposure, etc.), and emits an `Order` for those that pass.

4. **Orders → broker → deals**  
   - Execution submits orders to MT5 and tracks their lifecycle (pending, filled, cancelled, rejected).
   - MT5 generates deals and positions which are reconciled against expected internal state.

5. **Deals → performance metrics**  
   - `PerformanceTracker` reads closed deals from MT5 history, aggregates them into `DealRecord`s and `PerformanceReport`s, and exposes readiness gates for live trading.

6. **Monitoring & feedback**  
   - Logs, diagnostics CSVs, and feature snapshots feed the Streamlit dashboard and post‑session analysis tools.

---

## 3. Data & feature layer

### 3.1 RawDataStore

Although the implementation is not shown in full, usage patterns indicate that:

- `RawDataStore(root=settings.data.raw_dir)` encapsulates a directory tree of symbol/timeframe Parquet or CSV files.
- Main methods include:
  - `read(symbol, timeframe, end=None, as_polars=True)`: load the full history up to an optional `asof` time.
  - `read_last_n(symbol, timeframe, n)`: fast path for reading only the last N bars for live updates or short backtests.
  - `get_available_symbols()` / `get_available_timeframes(symbol)`: used by the dashboard to populate dropdowns.

The key invariant is that raw data is **immutable**. All derived features are recomputable from raw data and never written back into the raw store.

### 3.2 FeatureStore

`FeatureStore` is a Parquet‑based cache for feature DataFrames designed to avoid recomputing expensive indicators like EMA‑200, ATR‑14, or ADX‑14 on large histories:

- File layout (conceptually):
  - `data/features/{SYMBOL}/{TIMEFRAME}/v{CONFIG_HASH}{YYYY-MM}.parquet`
  - The `CONFIG_HASH` is an MD5 digest of the entire `PipelineConfig` dict. Any change to feature parameters or required columns invalidates previous caches for that hash.

- Major operations:
  - `write(symbol, timeframe, df, confighash)`: partition the DataFrame by month on `event_time` and write each partition as a Parquet file.
  - `read(symbol, timeframe, confighash, start=None, end=None)`: read and concatenate only those partitions whose month is within the requested time range; returns `None` if the cache is missing or stale.
  - `read_latest_row(...)`: fast path that returns just the last row, used for live trading feature snapshots.
  - `is_stale(symbol, timeframe, confighash, raw_store_max_time)`: determines whether cached features are behind the raw data by comparing their maximum `event_time`.
  - `invalidate(symbol, timeframe)` / `invalidate_all()`: clean up cached features.

This design cleanly separates **immutable raw data** from **recomputable derived features**, making cache invalidation simple and safe (“if in doubt, delete the cache; it will rebuild from raw”).

---

## 4. Feature engineering pipeline

### 4.1 PipelineConfig

`PipelineConfig` centralises all feature computation parameters so the rest of the system can remain largely configuration‑driven. Key groups:

- **Minimum history & rolling windows**
  - `min_bars` (e.g. 200) is driven by the longest rolling window needed (e.g. EMA‑200).

- **Technical indicators**
  - EMA periods, SMA periods.
  - `rsi_period`, `atr_period`, `bb_period`.
  - MACD `(fast, slow, signal)`, stochastic `(k, d)` parameters.
  - ADX period and optional VWAP inclusion.

- **ML features**
  - `return_horizons` for forward return labels and backward return features.
  - `vol_windows` for realised volatility windows.
  - `price_pos_windows` for price‑position in range indicators.
  - `momentum_windows` for momentum measures.

- **Target generation** (training only)
  - `add_targets` flag and `target_horizons` for generating forward return targets used by ML models.

- **Required feature columns**
  - `required_feature_cols` defines a minimal feature set (RSI, ATR, EMA, MACD, log returns, etc.) that must be non‑NaN for a row to be considered usable.

This centralisation prevents magic numbers from being scattered across strategy and training code and gives a single place to version the feature space.

### 4.2 FeaturePipeline lifecycle

`FeaturePipeline` orchestrates feature computation in five main steps:

1. **Load raw bars**  
   - `load_raw(symbol, timeframe, asof=None, nbars=None)` uses `RawDataStore` to fetch either the full history up to a timestamp or the last `n` bars.

2. **Validate and clean raw data**  
   - Deduplicate bars on `event_time` (keeping the last occurrence).
   - Sort ascending by `event_time` to ensure rolling windows are applied correctly.
   - Flag zero‑range bars (`high == low`) and zero‑volume bars, logging them as normal for gaps in some markets.
   - Compute single‑bar returns and log warnings for extreme jumps (e.g. >20% in one bar).
   - Detect time gaps by comparing successive `event_time` deltas to the median interval; large gaps are logged for awareness.

3. **Compute features**  
   - `add_all_technical(...)` adds indicator columns such as:
     - `ema_9`, `ema_21`, `ema_50`, `ema_200`, `sma_20`, `sma_50`.
     - `rsi_14`, `atr_14`.
     - Bollinger bands: `bb_mid_20`, `bb_upper_20`, `bb_lower_20`, `bb_width_20`, `%B`.
     - MACD line/signal/histogram.
     - Stochastic `%K` / `%D`.
     - ADX, `pdi_14`, `ndi_14`.
     - VWAP and distance to VWAP.
   - `add_all_ml_features(...)` adds ML‑oriented features and optionally target columns (`target_ret_{horizon}`).

4. **Filter & report**  
   - Count NaNs per column and drop any rows with NaNs in `required_feature_cols`, effectively discarding warm‑up periods where indicators have insufficient history.
   - Build a `DataQualityReport` with:
     - `total_bars`, `bars_after_filter`, `dropped_bars`, `drop_rate`.
     - `nan_counts` per column.
     - `gap_count` and human‑readable warnings about time gaps.
   - Log this report via loguru at INFO/WARNING levels.

5. **Cache (optional)**  
   - If a `FeatureStore` is configured, write the cleaned feature DataFrame, partitioned by month, keyed by the configuration hash.

### 4.3 compute_latest for live trading

For live trading, strategies do not need the full history; they require only a feature row at the current evaluation time with correct warm‑up:

- `compute_latest(symbol, timeframe, warmup_bars=300)` calls `compute(...)` with `nbars=warmup_bars` and returns `df.tail(1)` (a single row) or `None` if features are not ready.
- This method is used both by the main trading loop and by the dashboard’s “Feature Snapshot” panel.

---

## 5. Strategy framework

### 5.1 Direction and Signal

`Direction` is an `Enum` with three values:

- `LONG`: open or maintain a long position.
- `SHORT`: open or maintain a short position.
- `FLAT`: close an existing position (exit).

`Signal` is the atomic output of every strategy, with fields:

- `symbol`, `timeframe`: identify the instrument and bar timeframe.
- `direction`: one of `Direction`.
- `strength`: in `[0.0, 1.0]` – rule‑based strategies typically use `1.0`; ML strategies use class probability.
- `strategy_id`: unique identifier for the strategy instance.
- `bar_time`: event time of the bar that triggered the signal (UTC, timezone‑aware).
- `signal_time`: wall‑clock time at signal generation (UTC, timezone‑aware).
- `stop_loss`, `take_profit`: absolute price levels; `None` if they are delegated to the risk engine.
- `atr_at_signal`: ATR snapshot at signal time for volatility‑aware sizing.
- `meta`: free‑form dictionary for debug metadata.

Post‑init assertions ensure `strength` lies in `[0, 1]` and timestamps are UTC‑aware, which avoids subtle bugs in backtest vs live comparisons.

### 5.2 StrategyState

`StrategyState` is a simple serialisable state container that tracks, per strategy instance:

- `current_position`: one of `Direction` (default `FLAT`).
- `bars_in_trade`: number of bars elapsed since the last entry.
- `last_signal_time`: timestamp of the latest emitted signal.
- `total_signals`, `wins`, `losses`.
- A derived property `win_rate` computed as `wins / (wins + losses)`.

This state can be snapshotted and restored across process restarts to maintain continuity in long‑running live trading sessions.

### 5.3 BaseStrategy contract

`BaseStrategy` enforces a strict and testable contract:

- Subclasses must implement:
  - `on_bar(row: dict) -> Signal | None`: core decision logic given a single feature row.
  - `reset() -> None`: clear internal state prior to each backtest run or when re‑initialising.

- Subclasses may override:
  - `warmup_bars`: minimum number of bars required before any signal is emitted.
  - `on_fill(signal, fill_price)`: notification hook when a previously emitted signal has been executed.

- The engine calls `on_bar_update(row)` which wraps `on_bar` with:
  - Automatic bar counter increment.
  - Warm‑up guard: returns `None` until `bar_count >= warmup_bars`.
  - Signal counting: increments `total_signals` whenever a non‑`None` signal is returned.

- Helpers available to implementations:
  - `state`, `is_flat`, `is_long`, `is_short` properties for state inspection.
  - `_make_signal(row, direction, strength, stop_loss, take_profit, meta)` to generate consistent `Signal` objects.
  - `_get(row, key, default)` safe accessor for missing or `None` feature values.

The design explicitly disallows broker calls, external I/O, or global state inspection inside `on_bar`, making strategies easy to unit‑test and reuse across environments.

---

## 6. MeanReversionStrategy

`MeanReversionStrategy` is a concrete implementation of `BaseStrategy` that trades Bollinger Band mean reversion with RSI and ADX confirmation.

### 6.1 Intended behaviour

- **Entry (LONG)**  
  - Condition: `close < bb_lower` **and** `rsi < oversold_rsi` (e.g., 35).
  - Rationale: price is stretched below the lower band and momentum is oversold.

- **Entry (SHORT)**  
  - Condition: `close > bb_upper` **and** `rsi > overbought_rsi` (e.g., 65).
  - Rationale: price is stretched above the upper band and momentum is overbought.

- **Exit**  
  - Price crosses back through the mid‑band (mean reversion complete), or
  - RSI returns to neutral (~50), or
  - Maximum holding period in bars is exceeded.

The strategy is explicitly **counter‑trend** and is intended to operate in low‑ADX, ranging markets; it pairs naturally with a trend‑following strategy to build regime‑adaptive portfolios.

### 6.2 Safeguards

Several safeguards are coded in:

- **Band extremity**: requires close to be strictly outside the bands, not just touching them.
- **RSI confirmation**: avoids fading strong trends without momentum exhaustion.
- **ADX ceiling filter**: if `adx > adx_ceiling` (e.g., 30) the signal is skipped to avoid counter‑trading strong trends.
- **Minimum band width**: if `bb_width` is below `min_band_width`, the strategy does not trade because volatility is too low.
- **Max holding period**: after `max_bars_in_trade` bars, the trade is force‑closed.

### 6.3 Implementation highlights

- Required columns in the feature row:
  - `bb_lower_{bb_period}`, `bb_upper_{bb_period}`, `bb_mid_{bb_period}`, `bb_width_{bb_period}`.
  - `rsi_{rsi_period}`, `adx_{adx_period}`, `atr_{atr_period}`.
  - `close`, `event_time`.

- Entry logic sets:
  - Stop loss at `close ± atr * atr_multiplier` depending on direction.
  - Take profit at the Bollinger midline (`use_mid_as_tp=True`) or at a symmetric ATR multiple.

- Exit logic is direction‑dependent but symmetric:
  - For long positions: exit when `close >= mid` or `rsi > 50` or max bars exceeded.
  - For short positions: exit when `close <= mid` or `rsi < 50` or max bars exceeded.

Internal state stores entry price, stop loss, and take profit to maintain context across bars and is reset after each exit or when `reset()` is called.

---

## 7. Risk and cost engine

### 7.1 Order abstraction

`Order` models a single risk‑approved order ready for execution:

- **RiskEngine‑set fields**:
  - `signal`: originating `Signal`.
  - `lot_size`: approved position size in lots.
  - `order_type`: `MARKET`, `LIMIT`, or `STOP`.
  - `stop_loss`, `take_profit`: absolute SL/TP prices.
  - `risk_fraction`: fraction of equity at risk (for audit purposes).
  - `expected_cost`: estimated sum of spread + commission (and perhaps implied slippage).

- **Execution‑set fields**:
  - `order_id`: broker‑assigned ID.
  - `status`: `PENDING`, `SUBMITTED`, `FILLED`, `CANCELLED`, `REJECTED`.
  - `fill_price`: actual execution price.
  - `fill_time`: fill timestamp.
  - `meta`: free‑form dict for execution metadata.

- Convenience properties: `symbol`, `direction`, `strategy_id` delegate to the underlying `Signal`; `is_filled`, `is_entry`, `is_exit` expose common predicates.

### 7.2 CostFilter and edge vs cost

`CostFilter` is a quantitative gate that rejects trades where the expected edge is smaller than the transaction cost.

- **Inputs**:
  - `symbol`, `lot_size`.
  - `stop_distance` (in price units or points).
  - `rr_ratio`: expected reward:risk ratio.
  - `signal_strength`: strategy confidence in `[0,1]`.

- **Cost components**:
  - Spread cost: `(spread_points * pip_value) * lot_size`.
  - Commission cost: `commission_per_lot * lot_size` (round‑trip, account currency).
  - Slippage cost: `slippage_factor * spread_cost` to approximate additional adverse fill.

- **Edge estimate**:
  - Gross edge in account currency is `stop_distance * rr_ratio * pip_value * lot_size`.
  - Edge is then scaled by `signal_strength` to reflect ML probabilities.

- **Decision rule**:
  - Compute `edge_cost_ratio = edge_estimate / total_cost`.
  - Pass if `edge_cost_ratio >= min_edge_cost_ratio` (default 1.5).
  - Exit signals (with `stop_distance` `None` or non‑positive) bypass the filter.

`CostFilter.evaluate(...)` returns a `CostEstimate` with detailed breakdown of costs, edge, ratio, pass/fail flag, and a human‑readable `rejection_reason` which is logged at DEBUG level.

This design lets the system explicitly enforce “edge must be at least X× cost” across all strategies and symbols.

---

## 8. PerformanceTracker and readiness gates

### 8.1 DealRecord construction

`PerformanceTracker` integrates directly with MT5 via the `MetaTrader5` Python package to compute performance metrics based on **actual fills**, including slippage, spread, swap, and commission.

- It queries closed deals over a configurable look‑back period (`days_back`).
- Deals are grouped by `position_id` to unify entry and exit legs (and any intermediate partial fills) into a single logical trade.
- Each `DealRecord` stores:
  - `ticket`, `order`, `symbol`, `direction` (`"long"` or `"short"`).
  - `volume`, `open_price`, `close_price`.
  - `profit`, `swap`, `commission`.
  - `open_time`, `close_time`, `comment`.
  - Derived properties: `net_profit`, `is_win`, `duration_hours`.

### 8.2 PerformanceReport metrics

`PerformanceReport` aggregates trade‑level metrics for a given period (default 30 days):

- Trade counts: `total_trades`, `winning_trades`, `losing_trades`, `breakeven_trades`.
- P&L metrics:
  - `gross_profit`: sum of positive net profits.
  - `gross_loss`: absolute sum of negative net profits.
  - `total_net_pnl`: overall net P&L.
  - `total_commission`, `total_swap`.
- Ratios:
  - `win_rate` = `winning_trades / total_trades`.
  - `profit_factor` = `gross_profit / gross_loss` (∞ if no losses).
  - `expectancy`: average net P&L per trade.
  - `avg_win`, `avg_loss`, `largest_win`, `largest_loss`.
  - `avg_duration_hours`.
- Per‑symbol breakdown in `by_symbol`:
  - For each symbol: `trades`, symbol‑level `win_rate`, `net_pnl`, `avg_pnl`.

The `print_summary()` method prints a formatted report to the console, suitable for CLI workflows and quick inspection.

### 8.3 Live readiness check

`PerformanceTracker.is_ready_for_live(...)` encodes a quantitative gating rule for going live with real capital:

- Loads a `PerformanceReport` over a longer window (default 60 days).
- Applies a set of minimum thresholds:
  - `min_trades`: ensures sample size is statistically meaningful.
  - `min_win_rate`.
  - `min_expectancy` (in account currency per trade).
  - `min_profit_factor`.
- Builds a list of `readiness_notes` such as:
  - `✓ trades=123 >= 50` or `✗ win_rate=48.0% < 60.0%`.
- Returns `(ready_bool, report)`.

This gives both a pass/fail indicator (`ready_for_live`) and a human‑interpretable justification, which can be surfaced in ops dashboards or deployment runbooks.

---

## 9. Monitoring & observability

### 9.1 Streamlit dashboard

`monitoring/dashboard.py` is a Streamlit application providing a live trading dashboard.

- **Sidebar controls**:
  - Mode badge (`PAPER`, `LIVE`, `BACKTEST`) derived from `settings.mode`.
  - Symbol and timeframe selectors populated from `RawDataStore.get_available_symbols()` and `get_available_timeframes(symbol)`.
  - Auto‑refresh slider (`5–300` seconds) and a manual refresh button that clears Streamlit’s cache and reruns the app.
  - Last refresh timestamp (UTC).

- **Top panel – Account summary**:
  - Reads a simple `logs/equity_snapshot.txt` file written by the trading system.
  - Parses `equity`, `daily_pnl`, `daily_dd`, `peak_dd` and displays metrics with green/red drawdown badges.

- **Middle panels – Price chart and feature snapshot**:
  - Left: line chart of close prices for the selected symbol/timeframe (last ~200 bars) plus an OHLC table of the last 50 bars.
  - Right: “Feature Snapshot” with grouped technical features (`Trend`, `Momentum`, `Volatility`, `Returns`) showing numeric values for the latest bar.
  - Data is fetched via cached helper functions `_load_bars` and `_load_features`, which invoke `RawDataStore` and `FeaturePipeline.compute_latest`.

- **Bottom panels – Risk status and system info**:
  - `Risk Status` panel displays configured risk limits from `settings.risk` (max risk per trade, max exposure, daily/peak DD limits) and mode/server information.
  - `System` panel checks raw and feature stores:
    - Raw store: lists number of available symbols.
    - Feature store: summarises partitions and total cache size.
    - Recent logs: shows the last ~20 lines of the most recent `.log` file in the `logs` directory.

The dashboard runs an infinite auto‑refresh loop by updating `st.session_state.last_refresh`, sleeping for `refresh_interval` seconds, and then calling `st.rerun()`.

### 9.2 SignalDiagnostics

`SignalDiagnostics` provides a structured explanation of **why** a strategy did or did not fire on each evaluation.

- **Usage**:
  - Instantiated once: `diag = SignalDiagnostics(log_dir=Path("logs"))`.
  - Called on each evaluation: `diag.log_evaluation(strategy_id, strategy_type, symbol, timeframe, row, signal, extra_state)`.

- **Inputs captured**:
  - `bar_time`, `close`, `rsi_14`, `adx_14`, `ema_9`, `ema_21`, `bb_pct_b_20`, `atr_14` (when present).
  - `strategy_type` selector: `"trend"`, `"mean_reversion"`, or `"ml"`.

- **Diagnosis for trend strategies** (`_diagnose_trend`) includes:
  - Warm‑up messages if EMA or ADX are not yet computed.
  - Classification of market as `BULLISH` or `BEARISH` based on EMA spread.
  - Explanation when ADX is below the threshold (market ranging) or when no crossover has occurred.
  - Explicit messages like `CROSSOVER: BULLISH just occurred | adx=27.3 — signal should fire`.

- **Diagnosis for mean‑reversion strategies** (`_diagnose_mean_rev`) includes:
  - Warm‑up if Bollinger or RSI are not yet computed.
  - Skip reasons when ADX is above the ceiling or band width is too narrow.
  - Explanations for:
    - Price below lower band but RSI not oversold.
    - Price above upper band but RSI not overbought.
    - Price inside bands with `%B` and RSI values.
  - `SHOULD_FIRE` messages when both band and RSI conditions are aligned.

- **Outputs**:
  - Log line at DEBUG (no signal) or INFO (signal fired) level with:
    - Strategy ID, symbol/timeframe, bar time.
    - Key indicator values.
    - Signal summary (direction, strength) or `None`.
    - Diagnosis string.
  - CSV row appended to `logs/signal_diagnostics.csv` with the same fields, suitable for offline analysis and visualisation.

This module essentially turns “strategy returned `None`” from a black box into an interpretable state machine that can be inspected both during development and post‑session.

---

## 10. Backtesting and logs

While backtest orchestrator code is not fully shown, the following characteristics can be inferred from logs and design:

- Backtests reuse:
  - `RawDataStore` and `FeaturePipeline` for data and features.
  - The same `BaseStrategy` implementations used in live trading.
  - The same risk and cost model (`CostFilter`, position sizing, drawdown constraints) as live.

- Logs show sequences where the system:
  - Computes features for multiple symbols/timeframes at once.
  - Writes data quality reports and feature caches.
  - Performs periodic reconciliation against broker state.

Because features and strategies are shared, discrepancies between backtest and live mostly reduce to execution assumptions (fill prices, slippage, partial fills), which can be inspected via `PerformanceTracker` and diagnosis logs.

---

## 11. Configuration and settings

Although not shown in full, usage of `settings` across the code implies a central configuration module (`config.settings`) that encapsulates:

- `settings.mode`: `"paper"`, `"live"`, or `"backtest"`.
- `settings.data.raw_dir`, `settings.data.features_dir`: file system locations for data and feature caches.
- `settings.risk`: risk parameters surfaced in dashboards and used by risk engine components.
- `settings.mt5`: MT5 connection details (server, symbols, login credentials handled elsewhere).

Centralising these settings makes it straightforward to run the same codebase in multiple environments by swapping configuration files or environment variables.

---

## 12. Operational runbook (high‑level)

A typical operational flow for running this system might be:

1. **Initialisation**
   - Configure MT5 credentials and symbols in `settings.mt5`.
   - Configure data directories in `settings.data`.
   - Start the trading process (e.g., `python main.py --mode=paper`).

2. **During session**
   - Keep the Streamlit dashboard open to monitor equity, drawdown, risk status, and system logs.
   - Observe `SignalDiagnostics` output for strategy behaviour.
   - If performance is being evaluated for live readiness, periodically run `PerformanceTracker.print_live_readiness(...)`.

3. **End of session**
   - Inspect `signal_diagnostics.csv` and trading logs for anomalies.
   - Review `PerformanceReport` and per‑symbol breakdowns.
   - Adjust strategy parameters, risk settings, or symbol lists as required.

4. **Going live**
   - Use `PerformanceTracker.is_ready_for_live(...)` with conservative thresholds.
   - Switch `settings.mode` to `"live"` only when readiness criteria and risk requirements are satisfied.

---

## 13. Extension points

The architecture supports several natural extension points:

- **Adding new strategies**
  - Subclass `BaseStrategy`, implement `on_bar` and `reset`, and use `_make_signal` for consistent Signal creation.
  - Reuse existing features or extend `PipelineConfig` and feature modules.

- **ML‑based strategies**
  - Use `FeaturePipeline` with `add_targets=True` to generate training sets.
  - Train classification or regression models externally and wrap them in a `BaseStrategy` that maps feature rows to probabilistic `Signal`s (`strength` = class probability).

- **Risk engine enhancements**
  - Implement additional filters based on volatility regime, recent drawdown, or per‑strategy risk budgets.
  - Combine `CostFilter` with dynamic position sizing and daily/weekly risk caps.

- **Analytics and visualisation**
  - Build notebooks or dashboards on top of `signal_diagnostics.csv`, `PerformanceReport`s, and feature caches to analyse behaviour by regime, symbol, or timeframe.

Overall, the project exhibits a clean separation of concerns: data, features, strategy logic, risk, execution, and monitoring are distinct layers joined by narrow, well‑typed interfaces (`Signal`, `Order`, `DealRecord`), which makes the codebase amenable to extension and systematic testing.