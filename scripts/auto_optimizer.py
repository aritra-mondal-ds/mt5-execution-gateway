"""
scripts/auto_optimizer.py
--------------------------
Automatic strategy parameter optimization using Optuna.
Triggered after every TRIGGER_TRADES closed trades accumulate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

PARAMS_FILE    = _ROOT / "config" / "live_params.json"
TRIGGER_TRADES = 20
MIN_IMPROVEMENT = 0.10
N_TRIALS       = 30
LOOKBACK_DAYS  = 14


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return json.load(f)


def save_params(params: dict, reason: str) -> None:
    params["_last_updated"] = datetime.now(tz=timezone.utc).isoformat()
    params["_updated_by"]   = f"auto_optimizer: {reason}"
    params["_version"]      = params.get("_version", 0) + 1
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"AutoOptimizer: params saved (v{params['_version']}) — {reason}")


OUR_MAGIC_NUMBER = 234000  # MT5Broker magic number — only count OUR trades

def get_closed_trades(days_back: int = LOOKBACK_DAYS) -> list[dict]:
    try:
        import MetaTrader5 as mt5
        if mt5.terminal_info() is None:
            logger.warning("AutoOptimizer: MT5 not running — using empty trade list")
            return []
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days_back)
        raw   = mt5.history_deals_get(start, end)
        if not raw:
            return []

        # Filter to only trades placed by THIS system (magic number match)
        # This prevents the optimizer from learning the other bot's patterns
        our_deals = [d for d in raw if d.magic == OUR_MAGIC_NUMBER]
        total = len(raw)
        ours  = len(our_deals)
        logger.info(
            f"AutoOptimizer: {ours}/{total} deals match our magic={OUR_MAGIC_NUMBER} "
            f"({total-ours} from other systems — excluded)"
        )
        if not our_deals:
            logger.warning(
                "AutoOptimizer: no trades with our magic number found. "
                "System may not have executed any trades yet."
            )
            return []

        pos: dict[int, list] = {}
        for d in our_deals:
            pos.setdefault(d.position_id, []).append(d)
        trades = []
        for pid, deals in pos.items():
            deals = sorted(deals, key=lambda d: d.time)
            entry = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_IN), None)
            exit_ = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if not entry or not exit_:
                continue
            net = sum(d.profit + d.commission + d.swap for d in deals)
            trades.append({
                "symbol": entry.symbol,
                "direction": "long" if entry.type == mt5.DEAL_TYPE_BUY else "short",
                "net_pnl": net,
                "open_time": datetime.fromtimestamp(entry.time, tz=timezone.utc),
                "close_time": datetime.fromtimestamp(exit_.time, tz=timezone.utc),
            })
        return trades
    except Exception as e:
        logger.warning(f"AutoOptimizer: MT5 read failed: {e}")
        return []


def _expectancy(trades: list[dict]) -> float:
    if not trades:
        return -999.0
    return sum(t["net_pnl"] for t in trades) / len(trades)


def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t["net_pnl"] > 0) / len(trades)


def objective_trend(trial: "optuna.Trial", trades: list[dict]) -> float:
    adx_threshold  = trial.suggest_float("adx_threshold",  20.0, 40.0, step=1.0)
    atr_multiplier = trial.suggest_float("atr_multiplier",  1.0,  2.5, step=0.25)
    tp_multiplier  = trial.suggest_float("tp_multiplier",   2.0,  6.0, step=0.5)

    # Enforce minimum R:R of 2.0 — below this is mathematically unviable
    rr = tp_multiplier / atr_multiplier
    if rr < 2.0:
        return -999.0

    scored = []
    for t in trades:
        pnl = t["net_pnl"]
        if pnl > 0:
            scored.append(pnl)
        else:
            # Wider SL (higher atr_multiplier) reduces premature stop-outs
            scored.append(pnl * (1.0 / atr_multiplier))

    if not scored:
        return -999.0
    exp = sum(scored) / len(scored)
    wr  = sum(1 for p in scored if p > 0) / len(scored)
    return exp * (wr ** 0.5)


def objective_mean_rev(trial: "optuna.Trial", trades: list[dict]) -> float:
    oversold_rsi   = trial.suggest_float("oversold_rsi",   20.0, 45.0, step=1.0)
    overbought_rsi = trial.suggest_float("overbought_rsi", 55.0, 80.0, step=1.0)
    # FIX: raise adx_ceiling range to 50 to match live_params value of 45
    adx_ceiling    = trial.suggest_float("adx_ceiling",    20.0, 50.0, step=2.0)
    max_bars       = trial.suggest_int("max_bars_in_trade",  8,   48, step=4)

    scored = []
    for t in trades:
        hold = (t["close_time"] - t["open_time"]).total_seconds() / 3600
        pnl  = t["net_pnl"]
        if pnl > 0:
            speed_bonus = 1.0 + max(0, (max_bars - hold) / max_bars) * 0.2
            scored.append(pnl * speed_bonus)
        else:
            scored.append(pnl)

    if not scored:
        return -999.0
    exp = sum(scored) / len(scored)
    wr  = sum(1 for p in scored if p > 0) / len(scored)
    return exp * (wr ** 0.5)


def run_optimization(force: bool = False) -> bool:
    if not OPTUNA_AVAILABLE:
        logger.warning("AutoOptimizer: optuna not installed — skipping")
        return False

    logger.info("AutoOptimizer: starting optimization cycle")
    trades = get_closed_trades(LOOKBACK_DAYS)

    if not trades and not force:
        logger.info(f"AutoOptimizer: no closed trades in last {LOOKBACK_DAYS} days — skipping")
        return False
    if len(trades) < 5 and not force:
        logger.info(f"AutoOptimizer: only {len(trades)} trades — need at least 5 — skipping")
        return False

    current_params = load_params()
    current_exp    = _expectancy(trades)
    current_wr     = _win_rate(trades)

    logger.info(
        f"AutoOptimizer: {len(trades)} trades | "
        f"current expectancy={current_exp:.2f} | win_rate={current_wr:.1%}"
    )

    updated = False

    # Trend optimization — fixed seed ensures same data → same result
    if trades:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective_trend(trial, trades),
            n_trials=N_TRIALS, show_progress_bar=False,
        )
        best  = study.best_params
        best_score = study.best_value
        current_score = objective_trend(
            optuna.trial.FixedTrial({
                "adx_threshold":  float(current_params["trend"]["adx_threshold"]),
                "atr_multiplier": float(current_params["trend"]["atr_multiplier"]),
                "tp_multiplier":  float(current_params["trend"]["tp_multiplier"]),
            }),
            trades,
        )
        improvement = (best_score - current_score) / (abs(current_score) + 1e-9)
        logger.info(
            f"AutoOptimizer [trend]: best={best_score:.3f} "
            f"current={current_score:.3f} improvement={improvement:.1%}"
        )
        if improvement >= MIN_IMPROVEMENT:
            current_params["trend"]["adx_threshold"]  = round(best["adx_threshold"], 1)
            current_params["trend"]["atr_multiplier"] = round(best["atr_multiplier"], 2)
            current_params["trend"]["tp_multiplier"]  = round(best["tp_multiplier"], 1)
            updated = True
            logger.info(
                f"AutoOptimizer [trend]: DEPLOYED "
                f"adx={best['adx_threshold']:.1f} "
                f"atr={best['atr_multiplier']:.2f} tp={best['tp_multiplier']:.1f}"
            )
        else:
            logger.info("AutoOptimizer [trend]: no improvement — keeping current params")

    # Mean reversion optimization — fixed seed for reproducibility
    if trades:
        sampler2 = optuna.samplers.TPESampler(seed=42)
        study2 = optuna.create_study(direction="maximize", sampler=sampler2)
        study2.optimize(
            lambda trial: objective_mean_rev(trial, trades),
            n_trials=N_TRIALS, show_progress_bar=False,
        )
        best2 = study2.best_params
        best_score2 = study2.best_value
        current_score2 = objective_mean_rev(
            optuna.trial.FixedTrial({
                "oversold_rsi":      float(current_params["mean_reversion"]["oversold_rsi"]),
                "overbought_rsi":    float(current_params["mean_reversion"]["overbought_rsi"]),
                "adx_ceiling":       float(current_params["mean_reversion"]["adx_ceiling"]),
                "max_bars_in_trade": int(current_params["mean_reversion"]["max_bars_in_trade"]),
            }),
            trades,
        )
        improvement2 = (best_score2 - current_score2) / (abs(current_score2) + 1e-9)
        logger.info(
            f"AutoOptimizer [mean_rev]: best={best_score2:.3f} "
            f"current={current_score2:.3f} improvement={improvement2:.1%}"
        )
        if improvement2 >= MIN_IMPROVEMENT:
            current_params["mean_reversion"]["oversold_rsi"]      = round(best2["oversold_rsi"], 1)
            current_params["mean_reversion"]["overbought_rsi"]    = round(best2["overbought_rsi"], 1)
            current_params["mean_reversion"]["adx_ceiling"]       = round(best2["adx_ceiling"], 1)
            current_params["mean_reversion"]["max_bars_in_trade"] = int(best2["max_bars_in_trade"])
            updated = True
            logger.info(
                f"AutoOptimizer [mean_rev]: DEPLOYED "
                f"rsi_os={best2['oversold_rsi']:.1f} "
                f"rsi_ob={best2['overbought_rsi']:.1f} "
                f"adx_ceil={best2['adx_ceiling']:.1f}"
            )
        else:
            logger.info("AutoOptimizer [mean_rev]: no improvement — keeping current params")

    if updated:
        save_params(
            current_params,
            f"improved from {len(trades)} trades | exp={current_exp:.2f} -> optimized"
        )
    else:
        logger.info("AutoOptimizer: no params changed this cycle")

    return updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    updated = run_optimization(force=args.force)
    print(f"Optimization complete. Params updated: {updated}")