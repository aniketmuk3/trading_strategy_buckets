# region imports
from AlgorithmImports import *
from datetime import timedelta
# endregion


class PutHedgeSleeve:
    """
    #68 VIX-Ranked Conditional SPY Put Hedge — PSR 20.48.
    Maintains a SPY core position (sized to PSR weight) and overlays
    short-dated OTM puts when VIX rank > 0.50 over a 150-day lookback.
    """

    def __init__(self, algo, w, spy_security, vix_security):
        self._algo     = algo
        self._spy      = spy_security
        self._spy_sym  = spy_security.symbol
        self._vix      = vix_security

        # SPY core sized to PSR fraction; put overlay scales to that core
        self._spy_pct      = 0.90 * w
        self._options_alloc = 90        # shares per contract
        self._rank          = 0.0
        self._contract      = None
        self._days_exp      = 2
        self._dte           = 25
        self._otm           = 0.01
        self._lookback_iv   = 150
        self._iv_lvl        = 0.5

    def _calc_vix_rank(self):
        hist = self._algo.history(
            self._vix.symbol, self._lookback_iv, Resolution.DAILY)
        if hist.empty:
            return
        lo = hist["low"].min()
        hi = hist["high"].max()
        rng = hi - lo
        if rng == 0:
            return
        self._rank = (self._vix.price - lo) / rng

    def daily_rebalance(self):
        if self._algo.is_warming_up:
            return
        self._calc_vix_rank()

        # Maintain SPY core
        pv = float(self._algo.portfolio.total_portfolio_value)
        if pv > 0:
            cur_w = (self._algo.portfolio[self._spy_sym].holdings_value / pv)
            if abs(cur_w - self._spy_pct) > 0.02:
                self._algo.set_holdings(self._spy_sym, self._spy_pct)

        # Roll expiring contract
        if (self._contract
                and (self._contract.id.date - self._algo.time).days
                    <= self._days_exp):
            self._algo.liquidate(self._contract)
            self._contract = None

        # Buy put overlay when VIX is elevated
        if self._rank > self._iv_lvl:
            self._buy_put()

    def _buy_put(self):
        if not self._contract:
            self._contract = self._options_filter()
            return
        if not self._algo.portfolio[self._contract].invested:
            spy_qty = self._spy.holdings.quantity
            qty = round(spy_qty / self._options_alloc)
            if qty > 0:
                self._algo.buy(self._contract, qty)

    def _options_filter(self):
        chain = self._algo.option_chain(self._spy_sym, flatten=True).data_frame
        if chain is None or chain.empty:
            return None
        min_exp    = self._algo.time + timedelta(self._dte - 8)
        max_exp    = self._algo.time + timedelta(self._dte + 8)
        max_strike = self._spy.price * (1 - self._otm)
        chain = chain[
            (chain.expiry > min_exp)
            & (chain.expiry < max_exp)
            & (chain.right == OptionRight.PUT)
            & (chain.strike < max_strike)
        ]
        if chain.empty:
            return None
        expiry = (
            (chain.expiry - (self._algo.time + timedelta(self._dte)))
            .abs().sort_values().index[0].id.date
        )
        chain    = chain[chain.expiry == expiry]
        contract = (chain.strike - self._spy.price).abs().sort_values().index[0]
        self._algo.add_option_contract(contract)
        return contract
