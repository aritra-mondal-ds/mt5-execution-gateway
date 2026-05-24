"""
monitoring/performance_tracker.py
-----------------------------------
Reads MT5 closed deal history and computes real trading performance.

This is the source of truth for win rate measurement. It reads directly
from MT5's deal history — not simulated fills — so the numbers reflect
actual execution quality including slippage, spread, and swap.

Metrics computed:
  - Win rate (closed trades with profit > 0 / total closed trades)
  - Expectancy (average net P&L per trade)
  - Profit factor (gross profit / gross loss)
  - Average win / average loss
  - Sharpe-like ratio on trade returns
  - Per-symbol and per-direction breakdown

Usage:
    tracker = PerformanceTracker(broker)
    report = tracker.get_report(days_back=30)
    report.print_summary()

    # Check readiness for live trading
    if tracker.is_ready_for_live(min_trades=50, min_win_rate=0.60,
                                  min_expectancy=10.0):
        print("Ready to go live!")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# ---------------------------------------------------------------------------
# Deal record
# ---------------------------------------------------------------------------

@dataclass
class DealRecord:
    ticket: int
    order: int
    symbol: str
    direction: str          # "long" or "short"
    volume: float
    open_price: float
    close_price: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
    close_time: datetime
    comment: str = ""

    @property
    def net_profit(self) -> float:
        return self.profit + self.swap + self.commission

    @property
    def is_win(self) -> bool:
        return self.net_profit > 0

    @property
    def duration_hours(self) -> float:
        delta = self.close_time - self.open_time
        return delta.total_seconds() / 3600


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    period_days: int = 30

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    total_swap: float = 0.0

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0       # avg net P&L per trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Duration
    avg_duration_hours: float = 0.0

    # Per-symbol breakdown
    by_symbol: dict[str, dict] = field(default_factory=dict)

    # Readiness assessment
    ready_for_live: bool = False
    readiness_notes: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        bar = "─" * 52
        print(f"\n{bar}")
        print(f"  PERFORMANCE REPORT | last {self.period_days} days")
        print(f"  Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(bar)
        print(f"  Total trades      : {self.total_trades:>10}")
        print(f"  Winning trades    : {self.winning_trades:>10}")
        print(f"  Losing trades     : {self.losing_trades:>10}")
        print(f"  Win rate          : {self.win_rate:>10.2%}")
        print(bar)
        print(f"  Total net P&L     : {self.total_net_pnl:>+10.2f}")
        print(f"  Gross profit      : {self.gross_profit:>10.2f}")
        print(f"  Gross loss        : {self.gross_loss:>10.2f}")
        print(f"  Profit factor     : {self.profit_factor:>10.2f}")
        print(f"  Expectancy        : {self.expectancy:>+10.2f}  ← per trade avg")
        print(f"  Avg win           : {self.avg_win:>+10.2f}")
        print(f"  Avg loss          : {self.avg_loss:>+10.2f}")
        print(f"  Largest win       : {self.largest_win:>+10.2f}")
        print(f"  Largest loss      : {self.largest_loss:>+10.2f}")
        print(f"  Avg duration      : {self.avg_duration_hours:>9.1f}h")
        print(f"  Commission paid   : {self.total_commission:>10.2f}")
        print(bar)
        if self.by_symbol:
            print("  By symbol:")
            for sym, stats in sorted(self.by_symbol.items()):
                print(
                    f"    {sym:<10} trades={stats['trades']:>4} "
                    f"win={stats['win_rate']:.0%} "
                    f"pnl={stats['net_pnl']:>+8.2f}"
                )
        print(bar)
        status = "✅ READY" if self.ready_for_live else "⏳ NOT YET"
        print(f"  Live readiness    : {status}")
        for note in self.readiness_notes:
            print(f"    {'✓' if '✓' in note else '✗'} {note}")
        print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Reads MT5 deal history and computes performance metrics.

    Usage:
        tracker = PerformanceTracker()
        tracker.connect()                    # or pass MT5Broker

        report = tracker.get_report(days_back=30)
        report.print_summary()

        # Check specific thresholds
        ready = tracker.is_ready_for_live(
            min_trades=30,
            min_win_rate=0.55,
            min_expectancy=5.0,
            min_profit_factor=1.3,
        )
    """

    def __init__(self, magic_number: int | None = None) -> None:
        """
        magic_number: if set, only count deals placed by this EA.
                      None = count all deals on the account.
        """
        self._magic = magic_number
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize MT5 connection if not already connected."""
        if not MT5_AVAILABLE:
            logger.warning("PerformanceTracker: MetaTrader5 not installed")
            return False
        if mt5.terminal_info() is not None:
            self._connected = True
            return True
        logger.warning("PerformanceTracker: MT5 not running")
        return False

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def get_report(self, days_back: int = 30) -> PerformanceReport:
        """
        Fetch closed deals from MT5 and compute performance metrics.

        Args:
            days_back: how many calendar days of history to analyse

        Returns:
            PerformanceReport with all metrics populated.
        """
        report = PerformanceReport(period_days=days_back)
        deals = self._fetch_deals(days_back)

        if not deals:
            logger.info(
                f"PerformanceTracker: no closed deals in last {days_back} days"
            )
            return report

        logger.info(
            f"PerformanceTracker: analysing {len(deals)} deals "
            f"over last {days_back} days"
        )

        self._compute_metrics(deals, report)
        return report

    def is_ready_for_live(
        self,
        min_trades: int = 30,
        min_win_rate: float = 0.55,
        min_expectancy: float = 5.0,
        min_profit_factor: float = 1.3,
        days_back: int = 60,
    ) -> tuple[bool, PerformanceReport]:
        """
        Assess whether the system is ready to trade with real money.

        Returns (is_ready, report).

        Criteria (all must pass):
          - min_trades  : enough sample size to be statistically meaningful
          - min_win_rate: minimum fraction of winning trades
          - min_expectancy: minimum average P&L per trade (in account currency)
          - min_profit_factor: gross_profit / gross_loss >= threshold
        """
        report = self.get_report(days_back)
        notes = []
        passed = True

        # Sample size
        if report.total_trades >= min_trades:
            notes.append(f"✓ trades={report.total_trades} >= {min_trades}")
        else:
            notes.append(f"✗ trades={report.total_trades} < {min_trades} required")
            passed = False

        # Win rate
        if report.win_rate >= min_win_rate:
            notes.append(f"✓ win_rate={report.win_rate:.1%} >= {min_win_rate:.1%}")
        else:
            notes.append(f"✗ win_rate={report.win_rate:.1%} < {min_win_rate:.1%}")
            passed = False

        # Expectancy
        if report.expectancy >= min_expectancy:
            notes.append(f"✓ expectancy={report.expectancy:.2f} >= {min_expectancy}")
        else:
            notes.append(f"✗ expectancy={report.expectancy:.2f} < {min_expectancy}")
            passed = False

        # Profit factor
        if report.profit_factor >= min_profit_factor:
            notes.append(f"✓ profit_factor={report.profit_factor:.2f} >= {min_profit_factor}")
        else:
            notes.append(f"✗ profit_factor={report.profit_factor:.2f} < {min_profit_factor}")
            passed = False

        report.ready_for_live = passed
        report.readiness_notes = notes
        return passed, report

    def print_live_readiness(
        self,
        min_trades: int = 30,
        min_win_rate: float = 0.55,
        min_expectancy: float = 5.0,
        min_profit_factor: float = 1.3,
        days_back: int = 60,
    ) -> None:
        """Convenience: print full readiness assessment to console."""
        ready, report = self.is_ready_for_live(
            min_trades, min_win_rate, min_expectancy,
            min_profit_factor, days_back,
        )
        report.print_summary()

    # ------------------------------------------------------------------
    # MT5 data fetch
    # ------------------------------------------------------------------

    def _fetch_deals(self, days_back: int) -> list[DealRecord]:
        """
        Fetch closed deals from MT5 history.

        MT5 deal types:
          DEAL_TYPE_BUY  (0): buy deal (entry for long, exit for short)
          DEAL_TYPE_SELL (1): sell deal (entry for short, exit for long)

        We want to pair entry + exit deals into completed trades.
        MT5 groups these by position_id — each position has an open deal
        and a close deal with the same position_id.
        """
        if not MT5_AVAILABLE or mt5.terminal_info() is None:
            logger.warning("PerformanceTracker: MT5 not available — returning empty")
            return []

        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days_back)

        raw_deals = mt5.history_deals_get(start, end)
        if raw_deals is None or len(raw_deals) == 0:
            return []

        # Group by position_id to match open + close deals
        position_deals: dict[int, list] = {}
        for deal in raw_deals:
            pid = deal.position_id
            position_deals.setdefault(pid, []).append(deal)

        records = []
        for pid, deals in position_deals.items():
            # Filter by magic number if set
            if self._magic is not None:
                if not any(d.magic == self._magic for d in deals):
                    continue

            # Sort by time — first deal is entry, last is exit
            deals_sorted = sorted(deals, key=lambda d: d.time)

            # Find entry (IN) and exit (OUT) deals
            entry_deal = next(
                (d for d in deals_sorted if d.entry == mt5.DEAL_ENTRY_IN), None
            )
            exit_deal = next(
                (d for d in deals_sorted if d.entry == mt5.DEAL_ENTRY_OUT), None
            )

            if entry_deal is None or exit_deal is None:
                continue   # skip incomplete positions (still open)

            direction = (
                "long" if entry_deal.type == mt5.DEAL_TYPE_BUY else "short"
            )

            total_profit     = sum(d.profit for d in deals)
            total_swap       = sum(d.swap for d in deals)
            total_commission = sum(d.commission for d in deals)

            record = DealRecord(
                ticket=exit_deal.ticket,
                order=exit_deal.order,
                symbol=entry_deal.symbol,
                direction=direction,
                volume=entry_deal.volume,
                open_price=entry_deal.price,
                close_price=exit_deal.price,
                profit=total_profit,
                swap=total_swap,
                commission=total_commission,
                open_time=datetime.fromtimestamp(entry_deal.time, tz=timezone.utc),
                close_time=datetime.fromtimestamp(exit_deal.time, tz=timezone.utc),
                comment=exit_deal.comment or "",
            )
            records.append(record)

        logger.debug(f"PerformanceTracker: fetched {len(records)} completed trades")
        return records

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, deals: list[DealRecord], report: PerformanceReport
    ) -> None:
        net_pnls = [d.net_profit for d in deals]
        wins     = [p for p in net_pnls if p > 0]
        losses   = [p for p in net_pnls if p < 0]
        be       = [p for p in net_pnls if p == 0]

        report.total_trades    = len(deals)
        report.winning_trades  = len(wins)
        report.losing_trades   = len(losses)
        report.breakeven_trades = len(be)
        report.win_rate        = len(wins) / len(deals) if deals else 0.0

        report.gross_profit    = sum(wins) if wins else 0.0
        report.gross_loss      = abs(sum(losses)) if losses else 0.0
        report.total_net_pnl   = sum(net_pnls)
        report.total_commission = sum(d.commission for d in deals)
        report.total_swap      = sum(d.swap for d in deals)

        report.profit_factor = (
            report.gross_profit / report.gross_loss
            if report.gross_loss > 0 else float("inf")
        )
        report.expectancy = sum(net_pnls) / len(net_pnls) if net_pnls else 0.0
        report.avg_win    = sum(wins) / len(wins) if wins else 0.0
        report.avg_loss   = sum(losses) / len(losses) if losses else 0.0
        report.largest_win  = max(wins) if wins else 0.0
        report.largest_loss = min(losses) if losses else 0.0

        durations = [d.duration_hours for d in deals]
        report.avg_duration_hours = sum(durations) / len(durations) if durations else 0.0

        # Per-symbol breakdown
        symbols = set(d.symbol for d in deals)
        for sym in symbols:
            sym_deals = [d for d in deals if d.symbol == sym]
            sym_pnls  = [d.net_profit for d in sym_deals]
            sym_wins  = [p for p in sym_pnls if p > 0]
            report.by_symbol[sym] = {
                "trades":   len(sym_deals),
                "win_rate": len(sym_wins) / len(sym_deals) if sym_deals else 0.0,
                "net_pnl":  sum(sym_pnls),
                "avg_pnl":  sum(sym_pnls) / len(sym_pnls) if sym_pnls else 0.0,
            }