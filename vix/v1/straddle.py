# region imports
from AlgorithmImports import *
# endregion


class StradleSleeve:
    """
    #99 VIX Z-Score Straddle Mean Reversion.
    Short straddle when VIX z-score is high (mean reversion bet).
    Quantity scales with regime allocation weight and z-score magnitude.
    Hard cap at 2 contracts regardless of signal strength.

    Fix: portfolio.invested check is now SPX-option-specific,
    preventing SPY/OEF holdings from blocking straddle entry.
    """

    MAX_CONTRACTS = 2

    def __init__(self, algo, vix_security):
        self._algo = algo
        self._vix  = vix_security

        self._spx = algo.add_index_option("SPX")
        self._spx.set_filter(lambda u: u.straddle(30))

        period = 24 * 21   # ~2-year lookback
        self._vix.straddle_std = algo.std(
            self._vix.symbol, period, Resolution.DAILY)
        self._vix.straddle_sma = algo.sma(
            self._vix.symbol, period, Resolution.DAILY)

    def _spx_options_invested(self):
        """Check if we hold any SPX option positions specifically."""
        for kvp in self._algo.portfolio:
            sym = kvp.Key
            h   = kvp.Value
            if (h.invested
                    and sym.security_type == SecurityType.OPTION
                    and "SPX" in str(sym.underlying)):
                return True
        return False

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        if (not self._vix.straddle_std.is_ready
                or not self._vix.straddle_sma.is_ready):
            return

        z_score = (
            (self._vix.price - self._vix.straddle_sma.current.value)
            / self._vix.straddle_std.current.value
        )
        self._algo.plot("B_Straddle_Z", "VIX_Z", z_score)

        # Scale quantity by z-score magnitude and regime allocation
        raw_qty = -int(z_score)
        scale   = float(alloc_weight)
        if raw_qty > 0:
            quantity = min(max(1, int(raw_qty * scale * 2)), self.MAX_CONTRACTS)
        elif raw_qty < 0:
            quantity = max(min(-1, int(raw_qty * scale * 2)), -self.MAX_CONTRACTS)
        else:
            quantity = 0

        # No signal — liquidate any existing SPX option positions
        if not quantity:
            if self._spx_options_invested():
                self._algo.liquidate(self._spx.symbol)
            return

        # Already positioned in SPX options — don't re-enter
        if self._spx_options_invested():
            return

        chain = self._algo.current_slice.option_chains.get(
            self._spx.symbol, None)
        if not chain:
            return
        active = [c for c in chain if c.expiry > self._algo.time]
        if not active:
            return

        expiry = min(c.expiry for c in active)
        strike = sorted(
            [c for c in active if c.expiry == expiry],
            key=lambda x: abs(x.strike - self._spx.price)
        )[0].strike

        strategy = OptionStrategies.straddle(
            self._spx.symbol, strike, expiry)
        self._algo.order(strategy, quantity)

    def liquidate_short(self):
        """Liquidate short SPX option legs only (crisis override)."""
        for kvp in self._algo.portfolio:
            sym = kvp.Key
            h   = kvp.Value
            if (h.invested
                    and h.quantity < 0
                    and sym.security_type == SecurityType.OPTION
                    and "SPX" in str(sym.underlying)):
                self._algo.liquidate(sym)

    def on_assignment(self, order_event):
        if order_event.is_assignment:
            self._algo.liquidate(self._spx.symbol)
