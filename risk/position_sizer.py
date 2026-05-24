"""
risk/position_sizer.py
----------------------
ATR-based position sizing with optional Kelly fraction scaling.

Sizing formula (fixed fractional):
    risk_amount  = equity * max_risk_per_trade
    stop_distance = atr * atr_multiplier  (in price units)
    lot_size     = risk_amount / (stop_distance * pip_value_per_lot)

Kelly scaling (optional):
    full_kelly = edge / odds  (requires historical win_rate and avg_rr)
    fractional_kelly = full_kelly * kelly_fraction  (e.g. 0.25 = quarter Kelly)
    final_lot = lot_size * min(fractional_kelly, 1.0)

Notes:
    - pip_value_per_lot: for XAUUSD = 10 USD/lot/pip (standard lot, USD acct)
      For forex pairs it varies by quote currency.
      We use a configurable pip_value dict or a default.
    - lot_size is always clipped to [min_lot, max_lot].
    - max_lot enforces per-trade position limits independently of
      the account-wide exposure check in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


# ---------------------------------------------------------------------------
# Pip value defaults (USD account, standard lot = 100,000 units)
# Forex: pip = 0.0001, 1 lot = 10 USD/pip for most pairs
# XAUUSD: pip = 0.01, 1 lot = 100 oz, ~$10/pip at $1000 gold
# ---------------------------------------------------------------------------
_DEFAULT_PIP_VALUES: dict[str, float] = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 10.0,  # approx
    "AUDUSD": 10.0,
    "XAUUSD": 10.0,  # ~$10/pip per standard lot
    "BTCUSDT": 1.0,  # crypto: 1 unit = $1/pip (quote currency = USDT)
    "ETHUSDT": 1.0,
}


@dataclass
class SizingResult:
    lot_size: float
    risk_amount: float           # USD risked
    stop_distance: float         # in price units
    risk_fraction: float         # fraction of equity
    kelly_scale: float = 1.0     # Kelly multiplier applied
    clipped: bool = False        # True if lot_size hit min/max boundary
    rejection_reason: str = ""   # non-empty if sizing failed


class PositionSizer:
    """
    Calculates approved lot size for a signal given current account state.

    Usage:
        sizer = PositionSizer(config)
        result = sizer.calculate(
            equity=10_000,
            atr=15.0,
            symbol="XAUUSD",
            stop_distance=22.5,  # optional override; uses atr*multiplier if None
        )
        if result.rejection_reason:
            # sizing failed — don't trade
        else:
            order.lot_size = result.lot_size
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.01,    # 1% of equity
        atr_multiplier: float = 1.5,          # stop = ATR * multiplier
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        kelly_fraction: float = 0.0,          # 0 = disabled, 0.25 = quarter Kelly
        pip_values: dict[str, float] | None = None,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.kelly_fraction = kelly_fraction
        self._pip_values = pip_values or _DEFAULT_PIP_VALUES

    def calculate(
        self,
        equity: float,
        atr: float | None,
        symbol: str,
        stop_distance: float | None = None,  # override ATR-based stop
        win_rate: float | None = None,        # for Kelly (optional)
        avg_rr: float | None = None,          # avg reward:risk (for Kelly)
    ) -> SizingResult:
        """
        Calculate lot size. Returns SizingResult with rejection_reason
        set if sizing is impossible (e.g. ATR missing, equity zero).
        """
        if equity <= 0:
            return SizingResult(0, 0, 0, 0, rejection_reason="zero_equity")

        # Determine stop distance
        if stop_distance is None:
            if atr is None or atr <= 0:
                return SizingResult(0, 0, 0, 0, rejection_reason="atr_missing")
            stop_distance = atr * self.atr_multiplier

        if stop_distance <= 0:
            return SizingResult(0, 0, 0, 0, rejection_reason="zero_stop_distance")

        pip_value = self._pip_values.get(symbol, 10.0)
        risk_amount = equity * self.max_risk_per_trade
        lot_size = risk_amount / (stop_distance * pip_value)

        # Kelly scaling
        kelly_scale = 1.0
        if self.kelly_fraction > 0 and win_rate is not None and avg_rr is not None:
            kelly_scale = self._kelly_scale(win_rate, avg_rr)
            lot_size *= kelly_scale

        # Clip to bounds
        clipped = False
        if lot_size < self.min_lot:
            lot_size = self.min_lot
            clipped = True
        elif lot_size > self.max_lot:
            lot_size = self.max_lot
            clipped = True

        # Round to 2 decimal places (broker standard)
        lot_size = round(lot_size, 2)
        risk_fraction = (stop_distance * pip_value * lot_size) / equity

        logger.debug(
            f"PositionSizer | {symbol} | equity={equity:.0f} | "
            f"atr={atr} | stop_dist={stop_distance:.4f} | "
            f"lots={lot_size:.2f} | risk={risk_fraction:.2%}"
        )

        return SizingResult(
            lot_size=lot_size,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            risk_fraction=risk_fraction,
            kelly_scale=kelly_scale,
            clipped=clipped,
        )

    def _kelly_scale(self, win_rate: float, avg_rr: float) -> float:
        """
        Fractional Kelly multiplier.

        Full Kelly = (win_rate * avg_rr - loss_rate) / avg_rr
        Applied at self.kelly_fraction of full Kelly.
        Clamped to [0, 1] — never bet more than fixed fractional.
        """
        loss_rate = 1.0 - win_rate
        if avg_rr <= 0:
            return 0.0
        full_kelly = (win_rate * avg_rr - loss_rate) / avg_rr
        scaled = full_kelly * self.kelly_fraction
        return max(0.0, min(1.0, scaled))