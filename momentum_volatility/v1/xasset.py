# region imports
from AlgorithmImports import *
# endregion


class XAssetSleeve:
    """
    #226 Cross-Asset Time-Series Momentum Volatility-Weighted — PSR 19.56.
    Multi-horizon composite momentum across 12 ETFs; inverse-vol weighting
    of top 5. Total allocation scaled to PSR weight fraction W.
    """

    TICKERS = ["SPY", "IWM", "QQQ", "EFA", "EEM", "VNQ",
               "LQD", "GLD", "SHY", "IEF", "TLT", "AGG"]

    def __init__(self, algo, w):
        self._algo = algo
        self._w    = w          # PSR weight fraction
        periods    = [1, 3, 6, 12]

        self._securities = []
        for ticker in self.TICKERS:
            eq = algo.add_equity(
                ticker, Resolution.DAILY,
                data_normalization_mode=DataNormalizationMode.TOTAL_RETURN)
            eq.xa_indicators = [algo.momp(eq, p * 21) for p in periods]
            eq.xa_vol = IndicatorExtensions.of(
                StandardDeviation(3 * 21), algo.logr(eq, 1))
            self._securities.append(eq)

    def rebalance(self):
        if self._algo.is_warming_up:
            return
        ready = {
            sec: sum(i.current.value for i in sec.xa_indicators)
                 / len(sec.xa_indicators)
            for sec in self._securities
            if all(i.is_ready for i in sec.xa_indicators)
            and sec.xa_vol.is_ready
            and sec.xa_vol.current.value > 0
        }
        if not ready:
            return
        selected  = sorted(ready, key=lambda s: ready[s])[-5:]
        vol_sum   = sum(1 / s.xa_vol.current.value for s in selected)
        if vol_sum == 0:
            return
        targets = [
            PortfolioTarget(
                sec,
                (1 / sec.xa_vol.current.value / vol_sum) * self._w)
            for sec in selected
        ]
        # Liquidate de-selected positions
        selected_syms = {s.symbol for s in selected}
        for sec in self._securities:
            if (sec.symbol not in selected_syms
                    and self._algo.portfolio[sec.symbol].invested):
                self._algo.liquidate(sec.symbol)
        self._algo.set_holdings(targets)
