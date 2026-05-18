# region imports
from AlgorithmImports import *
import pandas as pd
# endregion


class ValMomSleeve:
    """
    #342 Large-Cap Value Momentum Long-Short — PSR 7.32.
    Dollar-neutral ranking on value (inverse P/B) + 12-month momentum.
    Allocation scaled to PSR weight fraction W.
    """

    def __init__(self, algo, w):
        self._algo         = algo
        self._w            = w
        self._mom_lookback = 12 * 22
        self._mom_delay    = 22
        self._universe     = []

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            if security.symbol.security_type != SecurityType.EQUITY:
                continue
            security.vm_consolidator = self._algo.resolve_consolidator(
                security, Resolution.DAILY)
            mom = RateOfChange(self._mom_lookback - self._mom_delay)
            security.vm_momentum = IndicatorExtensions.of(
                Delay(self._mom_delay), mom)
            self._algo.register_indicator(
                security, mom, security.vm_consolidator)
            for bar in self._algo.history[TradeBar](
                    security, self._mom_lookback + self._mom_delay):
                mom.update(bar.end_time, bar.close)
            if security not in self._universe:
                self._universe.append(security)

        for security in changes.removed_securities:
            if security in self._universe:
                self._algo.liquidate(security.symbol)
                self._algo.subscription_manager.remove_consolidator(
                    security, security.vm_consolidator)
                self._universe.remove(security)

    def rebalance(self):
        if self._algo.is_warming_up:
            return
        df = pd.DataFrame()
        for security in self._universe:
            if (not hasattr(security, "vm_momentum")
                    or not security.vm_momentum.is_ready):
                continue
            pb = security.fundamentals.valuation_ratios.pb_ratio
            if pb in (None, 0):
                continue
            df.loc[security, "Value"]    = 1 / pb
            df.loc[security, "Momentum"] = security.vm_momentum.current.value

        if df.empty:
            return

        # Dollar-neutral ranked weights
        ranked  = df.rank() - df.rank().mean()
        weights = ranked.mean(axis=1)
        weights /= abs(weights).sum()
        weights *= self._w          # scale to PSR share

        all_managed = {s for s in weights.index}
        for sec in self._universe:
            if (sec not in all_managed
                    and self._algo.portfolio[sec.symbol].invested):
                self._algo.liquidate(sec.symbol)

        for security, w in weights.items():
            self._algo.set_holdings(security.symbol, float(w))
