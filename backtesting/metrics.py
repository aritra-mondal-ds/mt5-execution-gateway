"""
backtesting/metrics.py
-----------------------
Performance metrics computed from a completed backtest's trade log
and equity curve.

All metrics are computed from the equity curve (list of equity values,
one per bar) and the trade log (list of closed trade dicts).

Metrics computed:
  From equity curve:
    - Total return
    - CAGR (Compound Annual Growth Rate)
    - Sharpe ratio (annualized, risk-free rate configurable)
    - Sortino ratio (downside deviation only)
    - Calmar ratio (CAGR / max drawdown)
    - Max drawdown (peak-to-trough, in %)
    - Max drawdown duration (bars)
    - Volatility (annualized std of returns)

  From trade log:
    - Win rate
    - Profit factor (gross profit / gross loss)
    - Average win / average loss
    - Expectancy (average P&L per trade)
    - Avg bars in trade
    - Max consecutive wins / losses
    - Total trades, longs, shorts

All inputs/outputs use plain Python floats and lists — no Polars/Pandas
dependency here so metrics can be computed incrementally during a live run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestMetrics:
    # Return metrics
    total_return: float = 0.0          # (final_equity / initial_equity) - 1
    cagr: float = 0.0                  # annualized return
    sharpe: float = 0.0                # annualized Sharpe ratio
    sortino: float = 0.0               # annualized Sortino ratio
    calmar: float = 0.0                # CAGR / max_drawdown

    # Risk metrics
    max_drawdown: float = 0.0          # max peak-to-trough as fraction
    max_drawdown_duration: int = 0     # bars spent in drawdown
    volatility: float = 0.0           # annualized return volatility

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0            # avg P&L per trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_bars_in_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    total_commission: float = 0.0

    # Equity curve
    initial_equity: float = 0.0
    final_equity: float = 0.0

    def __str__(self) -> str:
        return (
            f"{'─'*45}\n"
            f"  BACKTEST RESULTS\n"
            f"{'─'*45}\n"
            f"  Total return    : {self.total_return:>10.2%}\n"
            f"  CAGR            : {self.cagr:>10.2%}\n"
            f"  Sharpe ratio    : {self.sharpe:>10.2f}\n"
            f"  Sortino ratio   : {self.sortino:>10.2f}\n"
            f"  Calmar ratio    : {self.calmar:>10.2f}\n"
            f"  Max drawdown    : {self.max_drawdown:>10.2%}\n"
            f"  DD duration     : {self.max_drawdown_duration:>10} bars\n"
            f"  Volatility      : {self.volatility:>10.2%}\n"
            f"{'─'*45}\n"
            f"  Total trades    : {self.total_trades:>10}\n"
            f"  Win rate        : {self.win_rate:>10.2%}\n"
            f"  Profit factor   : {self.profit_factor:>10.2f}\n"
            f"  Expectancy      : {self.expectancy:>10.2f}\n"
            f"  Avg win         : {self.avg_win:>10.2f}\n"
            f"  Avg loss        : {self.avg_loss:>10.2f}\n"
            f"  Avg bars/trade  : {self.avg_bars_in_trade:>10.1f}\n"
            f"  Max consec wins : {self.max_consecutive_wins:>10}\n"
            f"  Max consec loss : {self.max_consecutive_losses:>10}\n"
            f"  Long / Short    : {self.long_trades:>4} / {self.short_trades:<4}\n"
            f"  Commission paid : {self.total_commission:>10.2f}\n"
            f"{'─'*45}\n"
            f"  Initial equity  : {self.initial_equity:>10.2f}\n"
            f"  Final equity    : {self.final_equity:>10.2f}\n"
            f"{'─'*45}"
        )


def compute_metrics(
    equity_curve: list[float],
    trade_log: list[dict],
    bars_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """
    Compute all performance metrics from equity curve and trade log.

    Args:
        equity_curve  : equity value at each bar, including initial value
        trade_log     : list of dicts with keys:
                        net_pnl, direction, bars_held, commission
        bars_per_year : 252 for daily, 252*6.5 for hourly equities, etc.
        risk_free_rate: annualized risk-free rate (default 0)

    Returns:
        BacktestMetrics dataclass with all computed values.
    """
    m = BacktestMetrics()

    if not equity_curve or len(equity_curve) < 2:
        return m

    m.initial_equity = equity_curve[0]
    m.final_equity = equity_curve[-1]

    # ── Return metrics ──────────────────────────────────────────────
    m.total_return = (m.final_equity / m.initial_equity) - 1.0

    n_bars = len(equity_curve) - 1
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0

    if years > 0 and m.initial_equity > 0:
        m.cagr = (m.final_equity / m.initial_equity) ** (1.0 / years) - 1.0

    # Bar-level returns
    returns = [
        (equity_curve[i] / equity_curve[i - 1]) - 1.0
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]

    if returns:
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0

        m.volatility = std_ret * math.sqrt(bars_per_year)

        # Sharpe ratio
        rf_per_bar = risk_free_rate / bars_per_year
        excess = [r - rf_per_bar for r in returns]
        excess_mean = sum(excess) / len(excess)
        excess_var = sum((r - excess_mean) ** 2 for r in excess) / len(excess)
        excess_std = math.sqrt(excess_var) if excess_var > 0 else 0.0
        m.sharpe = (excess_mean / excess_std * math.sqrt(bars_per_year)
                    if excess_std > 0 else 0.0)

        # Sortino ratio (downside deviation)
        downside = [r for r in returns if r < rf_per_bar]
        if downside:
            down_var = sum(r ** 2 for r in downside) / len(returns)
            down_std = math.sqrt(down_var)
            m.sortino = (excess_mean / down_std * math.sqrt(bars_per_year)
                         if down_std > 0 else 0.0)

    # ── Drawdown ────────────────────────────────────────────────────
    m.max_drawdown, m.max_drawdown_duration = _compute_drawdown(equity_curve)

    # Calmar ratio
    if m.max_drawdown > 0:
        m.calmar = m.cagr / m.max_drawdown
    elif m.cagr > 0:
        m.calmar = float("inf")

    # ── Trade metrics ───────────────────────────────────────────────
    if not trade_log:
        return m

    pnls = [t.get("net_pnl", 0.0) for t in trade_log]
    directions = [t.get("direction", "long") for t in trade_log]
    bars_held = [t.get("bars_held", 1) for t in trade_log]
    commissions = [t.get("commission", 0.0) for t in trade_log]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.total_trades = len(pnls)
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
    m.long_trades = sum(1 for d in directions if d == "long")
    m.short_trades = sum(1 for d in directions if d == "short")
    m.total_commission = sum(commissions)
    m.expectancy = sum(pnls) / len(pnls) if pnls else 0.0
    m.avg_win = sum(wins) / len(wins) if wins else 0.0
    m.avg_loss = sum(losses) / len(losses) if losses else 0.0
    m.avg_bars_in_trade = sum(bars_held) / len(bars_held) if bars_held else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Consecutive wins/losses
    m.max_consecutive_wins, m.max_consecutive_losses = _consecutive(pnls)

    return m


def _compute_drawdown(equity: list[float]) -> tuple[float, int]:
    """
    Returns (max_drawdown_fraction, max_drawdown_duration_bars).
    max_drawdown is expressed as a positive fraction (0.20 = 20% drawdown).
    """
    if not equity:
        return 0.0, 0

    peak = equity[0]
    max_dd = 0.0
    max_duration = 0
    dd_start = 0
    in_drawdown = False

    for i, eq in enumerate(equity):
        if eq > peak:
            peak = eq
            if in_drawdown:
                duration = i - dd_start
                max_duration = max(max_duration, duration)
            in_drawdown = False

        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                dd_start = i

    # Close any open drawdown at the end
    if in_drawdown:
        max_duration = max(max_duration, len(equity) - dd_start)

    return max_dd, max_duration


def _consecutive(pnls: list[float]) -> tuple[int, int]:
    """Returns (max_consecutive_wins, max_consecutive_losses)."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0

    for p in pnls:
        if p > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)

    return max_wins, max_losses