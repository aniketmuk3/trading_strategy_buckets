# region imports
from AlgorithmImports import *
from datetime import timedelta
# endregion


class PutHarvestSleeve:
    """
    #271 Systematic Short SPY Put Harvest.
    Sells ATM/ITM puts in calm VRP-positive regimes.
    Trade size scales with regime allocation weight.
    Gated externally: main.py checks vix_rank before calling sell_puts.
    Assignment handler liquidates any assigned SPY position immediately.
    """

    def __init__(self, algo, spy_security):
        self._algo    = algo
        self._spy     = spy_security
        self._spy_sym = spy_security.symbol
        self._min_expiry           = timedelta(40)
        self._max_expiry           = timedelta(55)
        self._max_trade_margin     = 0.20
        self._max_portfolio_margin = 0.50

    def sell_puts(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        try:
            chain   = self._algo.option_chain(self._spy)
            min_exp = self._algo.time + self._min_expiry
            max_exp = self._algo.time + self._max_expiry
            puts = [
                x for x in chain
                if (x.right == OptionRight.PUT
                    and min_exp < x.expiry < max_exp
                    and x.bid_price > 0
                    and x.volume > 0)
            ]
            if not puts:
                return

            furthest_expiry = max(x.expiry for x in puts)
            contract = sorted(
                [x for x in puts if x.expiry == furthest_expiry],
                key=lambda x: x.strike
            )[-1]

            # Scale trade margin by regime allocation weight
            effective_margin = self._max_trade_margin * float(alloc_weight)
            buying_power     = self._algo.portfolio.get_buying_power(
                self._spy_sym)
            unit_price = contract.strike * 100.0
            qty = int((buying_power * effective_margin) / unit_price)
            if not qty:
                return

            security = self._algo.add_option_contract(contract)

            # Portfolio margin safety check
            param = InitialMarginParameters(security, qty)
            initial_margin = self._spy.buying_power_model\
                .get_initial_margin_requirement(param).value
            denom = (self._algo.portfolio.total_margin_used
                     + self._algo.portfolio.margin_remaining)
            if denom == 0:
                return
            post_margin = (
                (initial_margin + self._algo.portfolio.total_margin_used)
                / denom)
            if post_margin > self._max_portfolio_margin:
                return

            self._algo.sell(contract, qty)
        except Exception:
            pass

    def liquidate_all(self):
        """Liquidate all short SPY put positions."""
        for kvp in self._algo.portfolio:
            sym = kvp.Key
            h   = kvp.Value
            if (h.invested
                    and h.quantity < 0
                    and sym.security_type == SecurityType.OPTION
                    and sym.underlying == self._spy_sym):
                self._algo.liquidate(sym)

    def on_assignment(self, assignment_event):
        """Liquidate any SPY shares received via assignment."""
        try:
            self._algo.market_order(
                assignment_event.symbol.underlying,
                -assignment_event.quantity * 100)
        except Exception:
            pass
