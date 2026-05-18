# region imports
from AlgorithmImports import *
from scipy import stats
import pandas as pd
# endregion


class MomVolSleeve:
    """
    #428 Monthly Large-Cap Momentum-Volatility LongShort — PSR 10.51.
    Cross-sectional z-score of momentum + volatility; equal-weight top/bottom 20.
    Allocation scaled to PSR weight fraction W.
    """

    def __init__(self, algo, w):
        self._algo      = algo
        self._w         = w
        self._lookback  = 252
        self._n_side    = 20
        self._universe  = []

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            sym = security.symbol
            # Skip instruments managed by other sleeves or non-equity
            if sym.security_type != SecurityType.EQUITY:
                continue
            security.mls_momentum = self._algo.roc(
                security, self._lookback - 1, Resolution.DAILY)
            self._algo.indicator_history(
                security.mls_momentum, security, self._lookback, Resolution.DAILY)
            security.mls_daily_ret  = self._algo.roc(security, 1, Resolution.DAILY)
            security.mls_volatility = IndicatorExtensions.of(
                StandardDeviation(self._lookback - 1), security.mls_daily_ret)
            self._algo.indicator_history(
                security.mls_daily_ret, security, self._lookback, Resolution.DAILY)
            if security not in self._universe:
                self._universe.append(security)

        for security in changes.removed_securities:
            if security in self._universe:
                self._algo.deregister_indicator(security.mls_momentum)
                self._algo.deregister_indicator(security.mls_daily_ret)
                self._universe.remove(security)

    def rebalance(self):
        if self._algo.is_warming_up:
            return
        securities = [
            s for s in self._universe
            if s.price
            and hasattr(s, "mls_momentum")
            and s.mls_momentum.is_ready
            and s.mls_volatility.is_ready
        ]
        if not securities:
            return
        factors = pd.DataFrame(
            {
                "momentum":   [s.mls_momentum.current.value   for s in securities],
                "volatility": [s.mls_volatility.current.value for s in securities],
            },
            index=securities,
        )
        scores = factors.apply(stats.zscore, ddof=0).sum(axis=1)
        n_side = min(self._n_side, int(len(securities) / 2))
        if n_side == 0:
            return
        w_long  =  self._w / n_side
        w_short = -self._w / n_side
        long_syms  = [s.symbol for s in scores.nlargest(n_side).index]
        short_syms = [s.symbol for s in scores.nsmallest(n_side).index]
        all_managed = set(long_syms) | set(short_syms)

        # Liquidate securities leaving the managed set
        for sec in self._universe:
            if (sec.symbol not in all_managed
                    and self._algo.portfolio[sec.symbol].invested):
                self._algo.liquidate(sec.symbol)

        for sym in long_syms:
            self._algo.set_holdings(sym, w_long)
        for sym in short_syms:
            self._algo.set_holdings(sym, w_short)
