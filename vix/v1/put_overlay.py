# region imports
from AlgorithmImports import *
from datetime import timedelta
# endregion


class PutOverlaySleeve:
    """
    #228 Long SPY VIX-Ranked Put Overlay.
    SPY core position scaled by regime allocation weight.
    Buys OTM puts when VIX rank > 0.5 over 150-day lookback.
    vix_rank is exposed publicly so PutHarvestSleeve can gate on it.
    Put is liquidated when VIX rank drops back below threshold
    to avoid unnecessary premium drag in calm regimes.
    """

    def __init__(self, algo, spy_security, vix_security):
        self._algo    = algo
        self._spy     = spy_security
        self._spy_sym = spy_security.symbol
        self._vix     = vix_security

        self._options_alloc = 90
        self._contract      = None
        self._days_exp      = 2
        self._dte           = 25
        self._otm           = 0.01
        self._lookback_iv   = 150
        self._iv_lvl        = 0.50
        self.vix_rank       = 0.0   # public: read by harvest gate

    def _calc_vix_rank(self):
        try:
            hist = self._algo.history(
                self._vix.symbol, self._lookback_iv, Resolution.DAILY)
            if hist.empty:
                return
            lo  = float(hist["low"].min())
            hi  = float(hist["high"].max())
            rng = hi - lo
            if rng == 0:
                return
            self.vix_rank = float((self._vix.price - lo) / rng)
        except Exception:
            pass

    def daily_rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        self._calc_vix_rank()

        # SPY core sized by regime allocation weight
        spy_target = float(alloc_weight) * 0.90
        pv = float(self._algo.portfolio.total_portfolio_value)
        if pv > 0:
            cur_w = float(
                self._algo.portfolio[self._spy_sym].holdings_value / pv)
            if abs(cur_w - spy_target) > 0.02:
                self._algo.set_holdings(self._spy_sym, spy_target)

        # Roll contract approaching expiry
        if (self._contract is not None and
                (self._contract.id.date - self._algo.time).days
                <= self._days_exp):
            self._algo.liquidate(self._contract)
            self._contract = None

        if self.vix_rank > self._iv_lvl:
            self._buy_put()
        else:
            # VIX calm — liquidate existing put to stop premium drag
            if (self._contract is not None
                    and self._algo.portfolio[self._contract].invested):
                self._algo.liquidate(self._contract)
                self._contract = None

    def _buy_put(self):
        if self._contract is None:
            self._contract = self._options_filter()
            return
        if not self._algo.portfolio[self._contract].invested:
            spy_qty = self._spy.holdings.quantity
            qty = round(spy_qty / self._options_alloc)
            if qty > 0:
                self._algo.buy(self._contract, qty)

    def _options_filter(self):
        try:
            chain = self._algo.option_chain(
                self._spy_sym, flatten=True).data_frame
            if chain is None or chain.empty:
                return None
            min_exp    = self._algo.time + timedelta(self._dte - 8)
            max_exp    = self._algo.time + timedelta(self._dte + 8)
            max_strike = float(self._spy.price) * (1 - self._otm)
            chain = chain[
                (chain.expiry > min_exp)
                & (chain.expiry < max_exp)
                & (chain.right == OptionRight.PUT)
                & (chain.strike < max_strike)
            ]
            if chain.empty:
                return None
            target_exp = self._algo.time + timedelta(self._dte)
            expiry = (
                (chain.expiry - target_exp).abs()
                .sort_values().index[0].id.date
            )
            chain    = chain[chain.expiry == expiry]
            contract = (
                (chain.strike - float(self._spy.price))
                .abs().sort_values().index[0]
            )
            self._algo.add_option_contract(contract)
            return contract
        except Exception:
            return None
