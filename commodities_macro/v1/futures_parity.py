# region imports
from AlgorithmImports import *
from datetime import timedelta
from math import log
import numpy as np
# endregion


class FuturesParitySleeve:
    """
    #233 Low-Vol Cross-Asset Futures Risk Parity + #26 Term Structure filter.

    Selects the 5 lowest-volatility futures from a diversified universe.
    Term structure signal (#26) acts as a tiebreaker: prefer backwardated
    contracts (positive roll return) over contango when vol is similar.

    Hard gates (checked via vix_gate_ok() in main.py):
      - VIX must be below its 1Y median
      - SPY must be above its 200-day SMA
    Gross exposure = 50% of sleeve allocation to limit crisis damage.

    Fixes applied:
    - Added timedelta and numpy imports (both missing)
    - Removed algo.set_warm_up(31) call — sleeve classes must not override
      the main algorithm's warmup period (was overriding 420d with 31d)
    - Fixed exchange check: replaced nonexistent
      algo.exchange_hours_is_open(fut) with fut.exchange.hours.is_open()
      via the private _exchange_open() helper, which is now actually called
    """

    TICKERS = [
        Futures.Indices.VIX,
        Futures.Indices.SP_500_E_MINI,
        Futures.Indices.NASDAQ_100_E_MINI,
        Futures.Indices.DOW_30_E_MINI,
        Futures.Energies.BRENT_CRUDE,
        Futures.Energies.GASOLINE,
        Futures.Energies.HEATING_OIL,
        Futures.Energies.NATURAL_GAS,
        Futures.Grains.CORN,
        Futures.Grains.OATS,
        Futures.Grains.SOYBEANS,
        Futures.Grains.WHEAT,
    ]

    def __init__(self, algo, spy_sym, vix_security):
        self._algo    = algo
        self._spy_sym = spy_sym
        self._vix     = vix_security
        self._futures = []
        for ticker in self.TICKERS:
            fut = algo.add_future(ticker, extended_market_hours=True)
            fut.vol = IndicatorExtensions.of(
                StandardDeviation(30), algo.roc(fut, 1, Resolution.DAILY))
            fut.set_filter(timedelta(0), timedelta(90))
            self._futures.append(fut)
        # NOTE: do NOT call algo.set_warm_up() here —
        # warmup is controlled exclusively by main.py (set to 420 days)

    def vix_gate_ok(self):
        """
        Returns True if VIX < 1Y median AND SPY > 200d SMA.
        Both conditions required for futures parity to activate.
        """
        try:
            vh = self._algo.history(
                [self._vix.symbol], 252, Resolution.DAILY)
            if vh.empty:
                return False
            vix_median = float(vh.loc[self._vix.symbol]["close"].median())
            vix_ok     = float(self._vix.price) < vix_median

            sh = self._algo.history(
                [self._spy_sym], 200, Resolution.DAILY)
            if sh.empty:
                return False
            spy_sma200 = float(sh.loc[self._spy_sym]["close"].mean())
            spy_price  = float(self._algo.securities[self._spy_sym].price)
            spy_ok     = spy_price > spy_sma200

            return vix_ok and spy_ok
        except Exception:
            return False

    def _exchange_open(self, fut):
        """Check if the future's exchange is currently open."""
        try:
            return fut.exchange.hours.is_open(self._algo.time, True)
        except Exception:
            return False

    def _roll_return(self, future):
        """
        Annualized roll return from near vs distant contract.
        Positive = backwardation, Negative = contango.
        """
        try:
            chain = self._algo.current_slice.future_chains.get(future.symbol)
            if chain is None or chain.contracts.count < 2:
                return 0.0
            contracts = sorted(list(chain), key=lambda c: c.expiry)
            near    = contracts[0]
            distant = contracts[-1]
            if near.expiry == distant.expiry:
                return 0.0
            p_near = (float(near.last_price) if near.last_price > 0
                      else 0.5 * float(near.ask_price + near.bid_price))
            p_dist = (float(distant.last_price) if distant.last_price > 0
                      else 0.5 * float(distant.ask_price + distant.bid_price))
            if p_near <= 0 or p_dist <= 0:
                return 0.0
            days = (distant.expiry - near.expiry).days
            return (log(p_near) - log(p_dist)) * 365.0 / days
        except Exception:
            return 0.0

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        w = float(alloc_weight)
        if w < 0.01:
            self.liquidate_all()
            return

        # Filter: mapped, vol ready, exchange open
        # Uses _exchange_open() — fixes the nonexistent method bug
        ready = [
            fut for fut in self._futures
            if (fut.mapped
                and fut.vol.is_ready
                and fut.vol.current.value > 0
                and self._exchange_open(fut))
        ]
        if not ready:
            return

        roll_returns = {fut: self._roll_return(fut) for fut in ready}

        def sort_key(fut):
            vol      = fut.vol.current.value
            roll     = roll_returns.get(fut, 0.0)
            discount = 0.02 * max(0.0, roll) if roll > 0 else 0.0
            return vol * (1.0 - discount)

        selected    = sorted(ready, key=sort_key)[:5]
        inv_vol_sum = sum(1.0 / f.vol.current.value for f in selected)
        if inv_vol_sum == 0:
            return

        # Gross exposure = 50% of sleeve allocation
        gross   = w * 0.50
        targets = [
            PortfolioTarget(
                fut.mapped,
                (1.0 / fut.vol.current.value / inv_vol_sum) * gross)
            for fut in selected
        ]
        self._algo.set_holdings(targets, True)

    def liquidate_all(self):
        for fut in self._futures:
            try:
                if fut.mapped and self._algo.portfolio[fut.mapped].invested:
                    self._algo.liquidate(fut.mapped)
            except Exception:
                pass
