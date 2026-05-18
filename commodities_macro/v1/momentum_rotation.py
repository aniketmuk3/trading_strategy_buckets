# region imports
from AlgorithmImports import *
# endregion


class MomentumRotationSleeve:
    """
    #276 Cross-Asset Momentum Rotation Vol-Scaled.
    Multi-horizon composite momentum across 12 ETFs.
    Selects top 5 by composite score, inverse-vol weighted.
    Total allocation scaled to regime weight.
    When regime shifts toward risk-off, this sleeve naturally
    rotates into TLT/IEF/GLD as equity momentum deteriorates.
    No bugs found — preserved as-is with minor safety guards added.
    """

    TICKERS = [
        "SPY", "IWM", "QQQ", "EFA", "EEM", "VNQ",
        "LQD", "GLD", "SHY", "IEF", "TLT", "AGG"
    ]

    def __init__(self, algo):
        self._algo       = algo
        self._securities = []
        for ticker in self.TICKERS:
            eq = algo.add_equity(ticker, Resolution.DAILY)
            eq.mom_indicators = [
                algo.momp(ticker, months * 21)
                for months in [1, 3, 6, 12]
            ]
            eq.mom_vol = IndicatorExtensions.of(
                StandardDeviation(3 * 21), algo.roc(ticker, 1))
            self._securities.append(eq)

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        w = float(alloc_weight)
        if w < 0.01:
            self._liquidate_all()
            return

        scored = {
            sec: sum(i.current.value for i in sec.mom_indicators)
            for sec in self._securities
            if (all(i.is_ready for i in sec.mom_indicators)
                and sec.mom_vol.is_ready
                and sec.mom_vol.current.value > 0)
        }
        if not scored:
            return

        selected = sorted(scored, key=lambda s: scored[s])[-5:]
        vol_sum  = sum(1.0 / s.mom_vol.current.value for s in selected)
        if vol_sum == 0:
            return

        targets = [
            PortfolioTarget(
                sec.symbol,
                (1.0 / sec.mom_vol.current.value / vol_sum) * w)
            for sec in selected
        ]
        selected_syms = {s.symbol for s in selected}
        for sec in self._securities:
            if (sec.symbol not in selected_syms
                    and self._algo.portfolio[sec.symbol].invested):
                self._algo.liquidate(sec.symbol)
        self._algo.set_holdings(targets)

    def _liquidate_all(self):
        for sec in self._securities:
            try:
                if self._algo.portfolio[sec.symbol].invested:
                    self._algo.liquidate(sec.symbol)
            except Exception:
                pass
