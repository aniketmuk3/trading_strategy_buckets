# region imports
from AlgorithmImports import *
# endregion


class AllWeatherSleeve:
    """
    #146 Levered Risk-Balanced Macro Diversifier.
    Fixed-weight all-weather allocation across equity, duration,
    and real assets at 1.5x leverage. Rebalanced annually plus
    whenever the monthly regime rebalance calls it.

    Allocation scaled by regime weight — acts as structural ballast,
    growing its share in risk-off and crisis regimes where the
    long-duration Treasury position (40% of sleeve) appreciates.
    Fixed weights within sleeve are preserved from original strategy.

    Fix: rebalance() is no longer passed as a direct schedule reference
    in main.py (would crash since it requires alloc_weight). Instead
    main.py calls it via _annual_rebalance() wrapper that fetches
    current regime allocations and passes the weight in.
    """

    WEIGHTS = {
        "VTI": 0.30,    # Total stock market
        "TLT": 0.40,    # 20+ year Treasuries (key risk-off anchor)
        "IEF": 0.15,    # 7-10 year Treasuries
        "GLD": 0.075,   # Gold
        "DBC": 0.075,   # Broad commodities
    }
    LEVERAGE = 1.5

    def __init__(self, algo):
        self._algo       = algo
        self._securities = {}
        for ticker, weight in self.WEIGHTS.items():
            eq = algo.add_equity(ticker, Resolution.DAILY)
            eq.target_weight = weight
            self._securities[ticker] = eq

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        w = float(alloc_weight)
        if w < 0.001:
            return
        targets = [
            PortfolioTarget(
                sec.symbol,
                sec.target_weight * self.LEVERAGE * w)
            for sec in self._securities.values()
        ]
        self._algo.set_holdings(targets)
