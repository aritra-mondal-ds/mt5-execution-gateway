"""
monitoring/signal_diagnostics.py
----------------------------------
Logs the exact feature values a strategy sees on every evaluation,
and explains why a signal was or wasn't generated.

Without this, a strategy returning None is a black box. With this,
every evaluation produces a structured snapshot showing:
  - Key indicator values at decision time
  - Which specific condition blocked the signal
  - The signal details if one was generated

Output goes to both loguru (DEBUG level) and a rotating CSV file
at logs/signal_diagnostics.csv for post-session analysis.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from strategies.base_strategy import Signal, Direction


# ---------------------------------------------------------------------------
# Condition checkers — explain WHY no signal
# ---------------------------------------------------------------------------

def _diagnose_trend(row: dict, fast_col: str, slow_col: str,
                    adx_col: str, adx_threshold: float,
                    prev_fast: float | None, prev_slow: float | None) -> str:
    """Return a human-readable string explaining trend strategy state."""
    fast  = row.get(fast_col)
    slow  = row.get(slow_col)
    adx   = row.get(adx_col)
    close = row.get("close")

    if fast is None or slow is None:
        return "WAITING: EMA not yet computed (warm-up)"
    if adx is None:
        return "WAITING: ADX not yet computed (warm-up)"

    spread = fast - slow
    trend_dir = "BULLISH" if spread > 0 else "BEARISH"

    if adx < adx_threshold:
        return (f"NO_SIGNAL: ADX={adx:.1f} < threshold={adx_threshold} "
                f"(market ranging) | spread={spread:.4f} ({trend_dir})")

    if prev_fast is None:
        return f"WAITING: no previous bar data yet | adx={adx:.1f}"

    prev_spread = prev_fast - prev_slow
    if prev_spread <= 0 < spread:
        return f"CROSSOVER: BULLISH just occurred | adx={adx:.1f} — signal should fire"
    if prev_spread >= 0 > spread:
        return f"CROSSOVER: BEARISH just occurred | adx={adx:.1f} — signal should fire"

    return (f"NO_SIGNAL: no crossover | "
            f"fast={fast:.4f} slow={slow:.4f} spread={spread:.4f} ({trend_dir}) | "
            f"adx={adx:.1f}")


def _diagnose_mean_rev(row: dict, bb_lower: str, bb_upper: str,
                       bb_mid: str, bb_width: str, rsi_col: str,
                       adx_col: str, adx_ceiling: float,
                       oversold: float, overbought: float) -> str:
    close  = row.get("close")
    lower  = row.get(bb_lower)
    upper  = row.get(bb_upper)
    mid    = row.get(bb_mid)
    width  = row.get(bb_width)
    rsi    = row.get(rsi_col)
    adx    = row.get(adx_col)

    if any(v is None for v in [close, lower, upper, rsi]):
        return "WAITING: features not yet computed (warm-up)"

    if adx is not None and adx > adx_ceiling:
        return (f"NO_SIGNAL: ADX={adx:.1f} > ceiling={adx_ceiling} "
                f"(strong trend, dangerous to counter-trade)")

    if width is not None and width < 0.001:
        return f"NO_SIGNAL: bands too tight (width={width:.4f})"

    if close < lower:
        if rsi < oversold:
            return f"SHOULD_FIRE: LONG setup | close={close:.4f} < lower={lower:.4f} | rsi={rsi:.1f} < {oversold}"
        else:
            return (f"NO_SIGNAL: price below band but RSI not oversold | "
                    f"close={close:.4f} lower={lower:.4f} rsi={rsi:.1f} (need < {oversold})")
    elif close > upper:
        if rsi > overbought:
            return f"SHOULD_FIRE: SHORT setup | close={close:.4f} > upper={upper:.4f} | rsi={rsi:.1f} > {overbought}"
        else:
            return (f"NO_SIGNAL: price above band but RSI not overbought | "
                    f"close={close:.4f} upper={upper:.4f} rsi={rsi:.1f} (need > {overbought})")
    else:
        pct_b = (close - lower) / (upper - lower) * 100 if upper != lower else 50
        return (f"NO_SIGNAL: price within bands ({pct_b:.0f}%B) | "
                f"close={close:.4f} lower={lower:.4f} upper={upper:.4f} | rsi={rsi:.1f}")


# ---------------------------------------------------------------------------
# Main diagnostics logger
# ---------------------------------------------------------------------------

class SignalDiagnostics:
    """
    Logs feature snapshots and signal decisions for every H1 evaluation.

    Usage (wired into main.py _on_bar):
        diag = SignalDiagnostics(log_dir=Path("logs"))
        diag.log_evaluation(
            strategy=strat,
            row=row,
            signal=signal_or_none,
            account_state=account_state,
        )
    """

    def __init__(
        self,
        log_dir: Path | str = Path("logs"),
        log_timeframes: list[str] | None = None,  # None = all, or ["H1", "H4"]
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._log_timeframes = log_timeframes  # None means log all timeframes
        self._csv_path = self._log_dir / "signal_diagnostics.csv"
        self._csv_initialized = False

    def log_evaluation(
        self,
        strategy_id: str,
        strategy_type: str,          # "trend" or "mean_reversion" or "ml"
        symbol: str,
        timeframe: str,
        row: dict,
        signal: Signal | None,
        extra_state: dict | None = None,   # e.g. prev_fast, prev_slow for trend
    ) -> None:
        """Log one strategy evaluation."""
        # Filter by timeframe if configured
        if self._log_timeframes and timeframe not in self._log_timeframes:
            return

        bar_time = row.get("event_time", "unknown")
        close    = row.get("close", 0)
        atr      = row.get("atr_14")
        rsi      = row.get("rsi_14")
        adx      = row.get("adx_14")
        ema9     = row.get("ema_9")
        ema21    = row.get("ema_21")
        bb_pctb  = row.get("bb_pct_b_20")

        # Build diagnosis string
        if strategy_type == "trend":
            state = extra_state or {}
            diagnosis = _diagnose_trend(
                row,
                fast_col=state.get("fast_col", "ema_9"),
                slow_col=state.get("slow_col", "ema_21"),
                adx_col="adx_14",
                adx_threshold=state.get("adx_threshold", 25.0),
                prev_fast=state.get("prev_fast"),
                prev_slow=state.get("prev_slow"),
            )
        elif strategy_type == "mean_reversion":
            state = extra_state or {}
            diagnosis = _diagnose_mean_rev(
                row,
                bb_lower="bb_lower_20", bb_upper="bb_upper_20",
                bb_mid="bb_mid_20", bb_width="bb_width_20",
                rsi_col="rsi_14", adx_col="adx_14",
                adx_ceiling=state.get("adx_ceiling", 30.0),
                oversold=state.get("oversold_rsi", 35.0),
                overbought=state.get("overbought_rsi", 65.0),
            )
        else:
            diagnosis = f"signal={'YES' if signal else 'NO'}"

        signal_str = (
            f"{signal.direction.value.upper()} strength={signal.strength:.2f}"
            if signal else "None"
        )

        # Pre-compute display strings — f-string format specs must be literals
        rsi_str  = f"{rsi:.1f}"  if rsi  is not None else "N/A"
        adx_str  = f"{adx:.1f}"  if adx  is not None else "N/A"
        ema9_str = f"{ema9:.4f}" if ema9 is not None else "N/A"
        ema21_str= f"{ema21:.4f}"if ema21 is not None else "N/A"

        # Log at INFO always for M15/H1 — visible in default log level
        # M1/M5 stay at DEBUG to avoid noise
        high_tf = timeframe in ("M15", "H1", "H4")
        log_fn = logger.info if (signal or high_tf) else logger.debug
        log_fn(
            f"DIAG | {strategy_id} | {symbol}/{timeframe} | {bar_time} | "
            f"close={close:.4f} rsi={rsi_str} adx={adx_str} "
            f"ema9={ema9_str} ema21={ema21_str} | "
            f"signal={signal_str} | {diagnosis}"
        )

        # Write to CSV
        self._write_csv({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "bar_time": str(bar_time),
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "close": round(close, 4) if close else "",
            "rsi_14": round(rsi, 2) if rsi else "",
            "adx_14": round(adx, 2) if adx else "",
            "ema_9": round(ema9, 4) if ema9 else "",
            "ema_21": round(ema21, 4) if ema21 else "",
            "bb_pct_b": round(bb_pctb, 3) if bb_pctb else "",
            "atr_14": round(atr, 4) if atr else "",
            "signal": signal_str,
            "diagnosis": diagnosis,
        })

    def _write_csv(self, row: dict) -> None:
        try:
            write_header = not self._csv_path.exists() or not self._csv_initialized
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    self._csv_initialized = True
                writer.writerow(row)
        except Exception as e:
            logger.debug(f"SignalDiagnostics CSV write failed: {e}")

    @property
    def csv_path(self) -> Path:
        return self._csv_path