# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
from sklearn.cluster import KMeans
# endregion


class VRPGateSleeve:
    """
    #70 SPX Put VRP Harvest Regime-Gated.
    Sells a single deep OTM SPX put only when both IV rank
    and strike availability are in low/moderate regimes
    (KMeans-classified). Internal dual-regime gate preserved exactly
    from original strategy.

    Fixes applied:
    - Added timedelta import
    - liquidate() calls now target the specific contract symbol,
      not the whole portfolio (avoids liquidating SPY/OEF positions)
    - portfolio.invested replaced with _spx_put_invested() check
    - rebalance() accepts alloc_weight directly (no separate set_alloc needed)
    """

    def __init__(self, algo):
        self._algo   = algo
        self._index  = algo.add_index("SPX")
        self._index.vrp_std = algo.std(
            self._index.symbol, 22, Resolution.DAILY)
        self._option = algo.add_index_option(self._index.symbol)
        self._option.set_filter(
            lambda u: u.include_weeklys().expiration(30, 90).strikes(-1, 1))
        self._option.iv_rank      = IVRank()
        self._option.strike_avail = StrikeAvailability()
        self._option.contract     = None
        self._current_alloc       = 0.069

    def _spx_put_invested(self):
        """Check if we specifically hold an SPX put position."""
        if self._option.contract is None:
            return False
        try:
            return self._algo.portfolio[self._option.contract].invested
        except Exception:
            return False

    def rebalance(self, alloc_weight=None):
        if alloc_weight is not None:
            self._current_alloc = float(alloc_weight)
        if self._algo.is_warming_up:
            return

        try:
            chain_df = self._algo.option_chain(
                self._index.symbol, flatten=True).data_frame
            if chain_df.empty:
                return
        except Exception:
            return

        # Update strike availability indicator
        if self._option.strike_avail.update(self._algo.time, chain_df):
            self._algo.plot("B_VRP_StrikeAvail", "Label",
                            self._option.strike_avail.label)

        # Update IV rank indicator
        universe_chain = self._algo.current_slice.option_chains.get(
            self._option.symbol)
        if not (universe_chain
                and self._option.iv_rank.update(universe_chain)):
            return

        self._algo.plot("B_VRP_IVRank", "Label",
                        self._option.iv_rank.label)

        # ── Exit conditions ───────────────────────────────────────────────────
        if self._spx_put_invested() and self._option.contract is not None:
            iv_high  = self._option.iv_rank.label == 2
            sa_high  = self._option.strike_avail.label == 2
            near_exp = (self._option.contract.id.date - self._algo.time
                        < timedelta(7))
            atm      = (self._index.price
                        <= self._option.contract.id.strike_price)

            if (iv_high and sa_high) or near_exp or atm:
                self._algo.liquidate(self._option.contract)
                self._option.contract = None
                return

        # ── Entry: both regimes must be low/moderate ──────────────────────────
        if (not self._spx_put_invested()
                and self._option.iv_rank.label < 2
                and self._option.strike_avail.label < 2):
            min_expiry = timedelta(30)
            filtered   = chain_df[
                (chain_df.expiry - self._algo.time >= min_expiry)
                & (chain_df.right == OptionRight.PUT)
                & (chain_df.strike <= self._index.price)
            ]
            if filtered.empty:
                return
            closest_exp = filtered.expiry.min()
            filtered    = filtered[
                filtered.expiry == closest_exp].sort_values("strike")
            std_val    = self._index.vrp_std.current.value
            strike_idx = -min(int(3 * std_val / 5), len(filtered))
            self._option.contract = filtered.index[strike_idx]
            self._algo.add_option_contract(self._option.contract)
            # Scale position size by regime allocation weight
            target_size = -0.25 * (self._current_alloc / 0.069)
            self._algo.set_holdings(self._option.contract, target_size)

    def liquidate_all(self):
        """Crisis override: liquidate SPX put contract specifically."""
        if self._option.contract is not None:
            try:
                self._algo.liquidate(self._option.contract)
            except Exception:
                pass
            self._option.contract = None


# ── Helper classes (preserved from original #70) ─────────────────────────────

class StrikeAvailability:
    def __init__(self, lookback=252, period=10):
        self._roc = RateOfChange(period)
        self._roc.window.size = lookback
        self._roc.window.reset()
        self.label    = 0
        self.value    = 0.0
        self.is_ready = False

    def update(self, t, chain):
        try:
            n_strikes = len(chain.strike.unique())
            ul_price  = float(chain.underlyinglastprice.iloc[0])
            if ul_price == 0:
                return False
            self._roc.update(t, n_strikes / ul_price)
            self.is_ready = self._roc.window.is_ready
            if self.is_ready:
                self.value   = self._roc.current.value
                window_vals  = np.array(
                    [x.value for x in self._roc.window][::-1]
                ).reshape(-1, 1)
                km = KMeans(
                    n_clusters=3, random_state=0, n_init=10
                ).fit(window_vals)
                centers   = km.cluster_centers_.ravel()
                label_map = {orig: srt for srt, orig
                             in enumerate(np.argsort(centers))}
                self.label = [label_map[l] for l in km.labels_][-1]
            return self.is_ready
        except Exception:
            return False


class IVRank:
    def __init__(self, lookback=252, min_expiry=30):
        self._min_iv  = Minimum(lookback)
        self._max_iv  = Maximum(lookback)
        self._min_exp = timedelta(min_expiry)
        self._history = RollingWindow[float](lookback)
        self.label    = 0
        self.value    = 0.0
        self.is_ready = False

    def update(self, chain):
        try:
            min_date  = chain.end_time + self._min_exp
            expiries  = [c.id.date for c in chain if c.id.date >= min_date]
            if not expiries:
                return False
            expiry    = min(expiries)
            contracts = [c for c in chain if c.id.date == expiry]
            abs_deltas = {
                c.symbol: abs(c.underlying_last_price - c.id.strike_price)
                for c in contracts
            }
            min_delta     = min(abs_deltas.values())
            atm_contracts = [c for c in contracts
                             if abs_deltas[c.symbol] == min_delta]
            agg_iv = float(np.median(
                [c.implied_volatility for c in atm_contracts]))
            self._history.add(agg_iv)
            self._min_iv.update(chain.end_time, agg_iv)
            self.is_ready = self._max_iv.update(chain.end_time, agg_iv)
            if self.is_ready:
                mn, mx     = (self._min_iv.current.value,
                              self._max_iv.current.value)
                self.value = float((agg_iv - mn) / (mx - mn + 1e-8))
                hist_arr   = np.array(
                    list(self._history)[::-1]).reshape(-1, 1)
                km = KMeans(
                    n_clusters=3, random_state=0, n_init=10
                ).fit(hist_arr)
                centers   = km.cluster_centers_.ravel()
                label_map = {orig: srt for srt, orig
                             in enumerate(np.argsort(centers))}
                self.label = [label_map[l] for l in km.labels_][-1]
            return self.is_ready
        except Exception:
            return False
