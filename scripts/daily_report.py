"""
scripts/daily_report.py
------------------------
Generates a daily trading summary printed to console and saved to
logs/daily_reports/YYYY-MM-DD.txt

Run manually or scheduled (e.g. via Windows Task Scheduler at 23:55 UTC):
    python scripts/daily_report.py

Sections:
  1. Account snapshot (equity, balance, open positions)
  2. Today's closed trades (P&L, win rate, expectancy)
  3. Running totals (cumulative since bot started)
  4. Strategy parameter history (what auto_optimizer changed)
  5. Signal diagnostics summary (why signals fired or didn't)
  6. Recommendation for tomorrow
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from loguru import logger

PARAMS_FILE    = _ROOT / "config" / "live_params.json"
DIAG_CSV       = _ROOT / "logs" / "signal_diagnostics.csv"
REPORT_DIR     = _ROOT / "logs" / "daily_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _get_account_snapshot() -> dict:
    try:
        import MetaTrader5 as mt5
        if mt5.terminal_info() is None:
            return {}
        info = mt5.account_info()
        positions = mt5.positions_get() or []
        return {
            "equity":    info.equity,
            "balance":   info.balance,
            "profit":    info.profit,
            "margin":    info.margin,
            "n_open":    len(positions),
            "open_syms": [p.symbol for p in positions],
        }
    except Exception as e:
        return {"error": str(e)}


def _get_todays_trades() -> list[dict]:
    try:
        import MetaTrader5 as mt5
        if mt5.terminal_info() is None:
            return []
        now   = datetime.now(tz=timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        raw   = mt5.history_deals_get(start, now)
        if not raw:
            return []

        pos: dict[int, list] = {}
        for d in raw:
            pos.setdefault(d.position_id, []).append(d)

        trades = []
        for pid, deals in pos.items():
            deals = sorted(deals, key=lambda d: d.time)
            entry = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_IN), None)
            exit_ = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if not entry or not exit_:
                continue
            net = sum(d.profit + d.commission + d.swap for d in deals)
            hold_h = (exit_.time - entry.time) / 3600
            trades.append({
                "symbol":    entry.symbol,
                "direction": "LONG" if entry.type == mt5.DEAL_TYPE_BUY else "SHORT",
                "lots":      entry.volume,
                "open":      entry.price,
                "close":     exit_.price,
                "net_pnl":   round(net, 2),
                "hold_h":    round(hold_h, 1),
                "win":       net > 0,
            })
        return trades
    except Exception:
        return []


def _get_all_time_trades(days: int = 60) -> list[dict]:
    try:
        import MetaTrader5 as mt5
        if mt5.terminal_info() is None:
            return []
        now   = datetime.now(tz=timezone.utc)
        start = now - timedelta(days=days)
        raw   = mt5.history_deals_get(start, now)
        if not raw:
            return []

        pos: dict[int, list] = {}
        for d in raw:
            pos.setdefault(d.position_id, []).append(d)

        trades = []
        for pid, deals in pos.items():
            deals = sorted(deals, key=lambda d: d.time)
            entry = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_IN), None)
            exit_ = next((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if not entry or not exit_:
                continue
            net = sum(d.profit + d.commission + d.swap for d in deals)
            trades.append({"net_pnl": net, "symbol": entry.symbol})
        return trades
    except Exception:
        return []


def _read_params() -> dict:
    try:
        return json.loads(PARAMS_FILE.read_text())
    except Exception:
        return {}


def _read_signal_diagnostics_today() -> dict:
    """Summarise today's signal_diagnostics.csv."""
    if not DIAG_CSV.exists():
        return {}
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "signals": 0, "reasons": defaultdict(int)})
    try:
        with open(DIAG_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("timestamp", "").startswith(today):
                    continue
                strat = row.get("strategy_id", "unknown")
                counts[strat]["total"] += 1
                if row.get("signal", "None") != "None":
                    counts[strat]["signals"] += 1
                diag = row.get("diagnosis", "")
                # Extract the first word as reason category
                reason = diag.split(":")[0].strip() if ":" in diag else diag[:20]
                counts[strat]["reasons"][reason] += 1
    except Exception:
        pass
    return dict(counts)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report() -> str:
    now    = datetime.now(tz=timezone.utc)
    lines  = []
    bar    = "═" * 60

    lines.append(bar)
    lines.append(f"  DAILY TRADING REPORT  —  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(bar)

    # ── 1. Account snapshot ───────────────────────────────────────────
    snap = _get_account_snapshot()
    lines.append("\n📊 ACCOUNT")
    if snap and "error" not in snap:
        lines.append(f"  Equity   : ${snap['equity']:>12,.2f}")
        lines.append(f"  Balance  : ${snap['balance']:>12,.2f}")
        lines.append(f"  Open P&L : ${snap['profit']:>+12,.2f}")
        lines.append(f"  Open pos : {snap['n_open']}  {snap['open_syms']}")
    else:
        lines.append("  MT5 not connected — restart bot to get live data")

    # ── 2. Today's trades ─────────────────────────────────────────────
    today_trades = _get_todays_trades()
    lines.append("\n📈 TODAY'S TRADES")
    if today_trades:
        wins   = [t for t in today_trades if t["win"]]
        losses = [t for t in today_trades if not t["win"]]
        total_pnl = sum(t["net_pnl"] for t in today_trades)
        win_rate  = len(wins) / len(today_trades)

        lines.append(f"  Trades   : {len(today_trades)}  "
                     f"(wins={len(wins)}, losses={len(losses)})")
        lines.append(f"  Win rate : {win_rate:.0%}")
        lines.append(f"  Total P&L: ${total_pnl:>+,.2f}")
        lines.append(f"  Avg P&L  : ${total_pnl/len(today_trades):>+,.2f}")
        lines.append("")
        for t in today_trades:
            emoji = "✅" if t["win"] else "❌"
            lines.append(
                f"  {emoji} {t['symbol']:8} {t['direction']:5} "
                f"{t['lots']:.2f}L  "
                f"open={t['open']:.4f} → {t['close']:.4f}  "
                f"P&L={t['net_pnl']:>+8.2f}  hold={t['hold_h']:.1f}h"
            )
    else:
        lines.append("  No closed trades today")

    # ── 3. Running totals ─────────────────────────────────────────────
    all_trades = _get_all_time_trades(days=60)
    lines.append("\n📉 RUNNING TOTALS (60 days)")
    if all_trades:
        pnls  = [t["net_pnl"] for t in all_trades]
        wins  = [p for p in pnls if p > 0]
        gl    = abs(sum(p for p in pnls if p < 0))
        gp    = sum(wins)
        lines.append(f"  Total trades : {len(pnls)}")
        lines.append(f"  Win rate     : {len(wins)/len(pnls):.0%}")
        lines.append(f"  Net P&L      : ${sum(pnls):>+,.2f}")
        lines.append(f"  Profit factor: {gp/gl:.2f}" if gl > 0 else "  Profit factor: ∞ (no losses)")
        lines.append(f"  Expectancy   : ${sum(pnls)/len(pnls):>+,.2f} per trade")
        # By symbol
        by_sym: dict[str, list] = defaultdict(list)
        for t in all_trades:
            by_sym[t["symbol"]].append(t["net_pnl"])
        lines.append("  By symbol:")
        for sym, sym_pnls in sorted(by_sym.items()):
            sym_wr = sum(1 for p in sym_pnls if p > 0) / len(sym_pnls)
            lines.append(
                f"    {sym:<10} {len(sym_pnls):>4} trades  "
                f"wr={sym_wr:.0%}  pnl=${sum(sym_pnls):>+,.2f}"
            )
    else:
        lines.append("  No completed trades yet")

    # ── 4. Strategy params ────────────────────────────────────────────
    params = _read_params()
    lines.append("\n⚙️  LIVE STRATEGY PARAMS")
    if params:
        lines.append(f"  Version      : v{params.get('_version', '?')}")
        lines.append(f"  Last updated : {params.get('_last_updated', 'never')}")
        lines.append(f"  Updated by   : {params.get('_updated_by', 'manual')}")
        t = params.get("trend", {})
        mr = params.get("mean_reversion", {})
        lines.append(f"  Trend        : adx>{t.get('adx_threshold')} "
                     f"ema{t.get('fast_period')}/{t.get('slow_period')} "
                     f"sl={t.get('atr_multiplier')}×ATR "
                     f"tp={t.get('tp_multiplier')}×ATR")
        lines.append(f"  MeanRev      : rsi<{mr.get('oversold_rsi')}/>{">"}{mr.get('overbought_rsi')} "
                     f"adx_ceil={mr.get('adx_ceiling')} "
                     f"max_hold={mr.get('max_bars_in_trade')}bars")

    # ── 5. Signal diagnostics ─────────────────────────────────────────
    diag = _read_signal_diagnostics_today()
    lines.append("\n🔍 SIGNAL DIAGNOSTICS TODAY")
    if diag:
        for strat, stats in diag.items():
            signal_rate = stats["signals"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(
                f"  {strat}: {stats['total']} evals → "
                f"{stats['signals']} signals ({signal_rate:.0%})"
            )
            top_reasons = sorted(
                stats["reasons"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            for reason, count in top_reasons:
                lines.append(f"    └ {reason}: {count}×")
    else:
        lines.append("  No diagnostic data yet (needs LOG_LEVEL=DEBUG or H1/H4 bars)")

    # ── 6. Recommendation ─────────────────────────────────────────────
    lines.append("\n💡 RECOMMENDATION FOR TOMORROW")
    if all_trades and len(all_trades) >= 10:
        pnls = [t["net_pnl"] for t in all_trades]
        wr   = sum(1 for p in pnls if p > 0) / len(pnls)
        exp  = sum(pnls) / len(pnls)
        if exp > 20 and wr > 0.60:
            lines.append("  ✅ System performing well. No changes needed.")
        elif exp > 0 and wr > 0.50:
            lines.append("  ⚡ Positive but room to improve. Run auto_optimizer.py.")
        elif len(all_trades) < 20:
            lines.append("  ⏳ Not enough trades for statistical significance. Keep running.")
        else:
            lines.append("  ⚠️  Below target. Run: python scripts/auto_optimizer.py --force")
    else:
        lines.append("  ⏳ Accumulating trade data. System needs 10+ trades for assessment.")

    lines.append(f"\n{bar}\n")
    return "\n".join(lines)


def save_report(text: str) -> Path:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path  = REPORT_DIR / f"{today}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = build_report()
    print(report)
    path = save_report(report)
    print(f"Report saved: {path}")