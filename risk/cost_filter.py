"""
risk/cost_filter.py
--------------------
Rejects signals where the expected edge is smaller than transaction cost.

Problem this solves:
  A strategy may find a valid signal with a small expected edge.
  If spread + commission exceeds that edge, the trade is net-negative
  in expectation even before slippage. The cost filter is the last
  quantitative gate before order submission.

Cost components modelled:
  1. Spread cost : (spread_points * pip_value) * lot_size
  2. Commission  : commission_per_lot * lot_size  (round-trip)
  3. Slippage    : slippage_factor * spread_cost  (market order only)

Edge estimate:
  edge_pips = (stop_distance * rr_ratio) * signal_strength
  edge_usd  = edge_pips * pip_value * lot_size

Filter condition:
  PASS if edge_usd / total_cost >= min_edge_cost_ratio (default 1.5)
  i.e. expected return must be at least 1.5x the cost to enter.

If signal has no stop_loss (exit signal), cost filter is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger


_DEFAULT_SPREADS: dict[str, float] = {
    "EURUSD": 1.0,    # ~1 pip
    "GBPUSD": 1.5,
    "USDJPY": 1.0,
    "XAUUSD": 30.0,   # ~30 points ($0.30)
    "BTCUSDT": 10.0,  # $10 spread
    "ETHUSDT": 1.0,
}

_DEFAULT_PIP_VALUES: dict[str, float] = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 10.0,
    "XAUUSD": 10.0,
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.0,
}


@dataclass
class CostEstimate:
    spread_cost: float
    commission_cost: float
    slippage_cost: float
    total_cost: float
    edge_estimate: float
    ratio: float              # edge / cost
    passed: bool
    rejection_reason: str = ""


class CostFilter:
    """
    Estimates transaction cost and compares against expected edge.

    Usage:
        filt = CostFilter(min_edge_cost_ratio=1.5)
        estimate = filt.evaluate(
            symbol="XAUUSD",
            lot_size=0.1,
            stop_distance=22.5,
            rr_ratio=2.0,
            signal_strength=0.8,
        )
        if not estimate.passed:
            # reject signal
    """

    def __init__(
        self,
        min_edge_cost_ratio: float = 1.5,
        commission_per_lot: float = 7.0,      # round-trip USD per lot
        slippage_factor: float = 0.5,          # 50% of spread as extra slippage
        spread_overrides: dict[str, float] | None = None,
        pip_value_overrides: dict[str, float] | None = None,
    ) -> None:
        self.min_ratio = min_edge_cost_ratio
        self.commission_per_lot = commission_per_lot
        self.slippage_factor = slippage_factor
        self._spreads = {**_DEFAULT_SPREADS, **(spread_overrides or {})}
        self._pip_values = {**_DEFAULT_PIP_VALUES, **(pip_value_overrides or {})}

    def evaluate(
        self,
        symbol: str,
        lot_size: float,
        stop_distance: float | None,
        rr_ratio: float = 2.0,
        signal_strength: float = 1.0,
    ) -> CostEstimate:
        """
        Estimate cost vs edge and return a CostEstimate.

        Args:
            symbol          : instrument
            lot_size        : approved lot size from PositionSizer
            stop_distance   : distance to stop in price units (None → skip)
            rr_ratio        : expected reward:risk ratio
            signal_strength : confidence in [0,1] from Signal.strength
        """
        # Exit signals have no SL — skip cost filter
        if stop_distance is None or stop_distance <= 0:
            return CostEstimate(
                spread_cost=0, commission_cost=0, slippage_cost=0,
                total_cost=0, edge_estimate=0, ratio=999.0, passed=True,
            )

        pip_value = self._pip_values.get(symbol, 10.0)
        spread = self._spreads.get(symbol, 2.0)

        # Cost calculation
        spread_cost     = spread * pip_value * lot_size
        commission_cost = self.commission_per_lot * lot_size
        slippage_cost   = spread_cost * self.slippage_factor
        total_cost      = spread_cost + commission_cost + slippage_cost

        # Edge estimate: discounted by signal confidence
        gross_edge   = stop_distance * rr_ratio * pip_value * lot_size
        edge_estimate = gross_edge * signal_strength

        ratio = edge_estimate / total_cost if total_cost > 0 else 0.0
        passed = ratio >= self.min_ratio

        rejection_reason = ""
        if not passed:
            rejection_reason = (
                f"edge_cost_ratio={ratio:.2f} < {self.min_ratio} | "
                f"edge={edge_estimate:.2f} cost={total_cost:.2f}"
            )
            logger.debug(
                f"CostFilter REJECT | {symbol} | {rejection_reason}"
            )
        else:
            logger.debug(
                f"CostFilter PASS | {symbol} | "
                f"ratio={ratio:.2f} edge={edge_estimate:.2f} cost={total_cost:.2f}"
            )

        return CostEstimate(
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            edge_estimate=edge_estimate,
            ratio=ratio,
            passed=passed,
            rejection_reason=rejection_reason,
        )