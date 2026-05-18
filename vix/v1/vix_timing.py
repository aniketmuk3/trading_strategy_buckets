# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class VIXTimingSleeve:
    """
    #49 VIX Predicts Stock Index Returns.
    Contrarian directional tilt on OEF using VIX percentile rank.
    Buy when VIX is in upper tail (fear peak → forward return positive).
    Short/flat when VIX is in lower tail (complacency → adverse).
    Position size scales with regime allocation weight.
    Given weak standalone PSR (1.03), acts as a small macro tilt only.
    Short side sized at 50% of long to reflect asymmetric confidence.
    """

    def __init__(self, algo, vix_security, oef_sym):
        self._algo       = algo
        self._vix        = vix_security
        self._oef_sym    = oef_sym
        self._lookback   = 504   # ~2 years
        self._long_pct   = 90
        self._short_pct  = 10
        self._window     = RollingWindow[float](self._lookback)

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return

        price = float(self._vix.price)
        self._window.add(price)
        if not self._window.is_ready:
            return

        history = list(self._window)
        w       = float(alloc_weight)

        if price > np.percentile(history, self._long_pct):
            # Fear peak — go long OEF scaled to allocation weight
            self._algo.set_holdings(self._oef_sym, w)
        elif price < np.percentile(history, self._short_pct):
            # Complacency — small short; 50% of long size given weak PSR
            self._algo.set_holdings(self._oef_sym, -w * 0.5)
        else:
            # No extreme signal — flat
            if self._algo.portfolio[self._oef_sym].invested:
                self._algo.liquidate(self._oef_sym)
