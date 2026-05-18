# region imports
from AlgorithmImports import *
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
# endregion


class MacroRotationSleeve:
    """
    #72 Monthly Macro Factor Cross-Asset Rotation — ML anchor.
    Decision Tree Regressor retrained monthly on VIX, yield curve,
    and fed funds rate to predict 21-day forward returns across
    SPY, GLD, BND, BTCUSD. Allocates to assets with positive
    predicted returns, proportional to predicted magnitude.
    Gross exposure = 1.5x sleeve allocation. BTC cap = 5%.

    Fixes applied:
    - Added timedelta and pandas imports (both were missing)
    - Split set_holdings into equity targets + separate BTC market_order
      because QC's set_holdings does not support mixed Equity + Crypto
      security types in a single call (raises Unsupported security type)
    """

    def __init__(self, algo):
        self._algo     = algo
        self._bitcoin  = algo.add_crypto(
            "BTCUSD", market=Market.BITFINEX, leverage=2).symbol
        self._equities = [
            algo.add_equity(t).symbol for t in ["SPY", "GLD", "BND"]]
        self._symbols  = self._equities + [self._bitcoin]
        self._factors  = [
            algo.add_data(Fred, t, Resolution.DAILY).symbol
            for t in ["VIXCLS", "T10Y3M", "DFF"]
        ]
        self._model    = DecisionTreeRegressor(max_depth=12, random_state=1)
        self._scaler   = StandardScaler()
        self._lookback = timedelta(4 * 365)
        self._btc_cap  = 0.05
        self._leverage = 1.5

    def rebalance(self, alloc_weight):
        if self._algo.is_warming_up:
            return
        w = float(alloc_weight)
        if w < 0.01:
            self._liquidate()
            return

        try:
            factors = self._algo.history(
                self._factors, self._lookback, Resolution.DAILY)
            if factors.empty:
                return
            factors = factors["value"].unstack(0).dropna()
        except Exception:
            return

        try:
            labels_raw = self._algo.history(
                self._symbols, self._lookback, Resolution.DAILY,
                data_normalization_mode=DataNormalizationMode.TOTAL_RETURN)
            if labels_raw.empty:
                return
            labels = (labels_raw["close"].unstack(0)
                      .dropna().pct_change(21).shift(-21).dropna())
        except Exception:
            return

        predictions = {}
        for symbol in self._symbols:
            if symbol not in labels.columns:
                continue
            asset_labels = labels[symbol].dropna()
            idx = factors.index.intersection(asset_labels.index)
            if len(idx) < 50:
                continue
            try:
                self._model.fit(
                    self._scaler.fit_transform(factors.loc[idx]),
                    asset_labels.loc[idx])
                pred = self._model.predict(
                    self._scaler.transform([factors.iloc[-1]]))[0]
                if pred > 0:
                    predictions[symbol] = float(pred)
            except Exception:
                continue

        if not predictions:
            self._liquidate()
            return

        pred_series = pd.Series(predictions)
        total_pred  = pred_series.sum()
        if total_pred == 0:
            self._liquidate()
            return
        weights = self._leverage * w * pred_series / total_pred

        # Cap BTC weight
        if self._bitcoin in weights.index:
            btc_cap = self._btc_cap * w
            if weights[self._bitcoin] > btc_cap:
                excess = weights[self._bitcoin] - btc_cap
                weights[self._bitcoin] = btc_cap
                eq_syms = [s for s in self._equities if s in weights.index]
                if eq_syms:
                    eq_w     = weights[eq_syms]
                    eq_total = eq_w.sum()
                    if eq_total > 0:
                        weights[eq_syms] = eq_w + excess * (eq_w / eq_total)

        # Liquidate positions no longer in target set
        target_syms = set(weights.index)
        for sym in self._symbols:
            if (sym not in target_syms
                    and self._algo.portfolio[sym].invested):
                self._algo.liquidate(sym)

        # ── set_holdings does NOT support mixed Equity + Crypto in one call ──
        # Split into two separate operations to avoid runtime error.

        # Equity targets (SPY, GLD, BND)
        equity_targets = [
            PortfolioTarget(sym, float(wt))
            for sym, wt in weights.items()
            if sym != self._bitcoin
        ]
        if equity_targets:
            self._algo.set_holdings(equity_targets)

        # BTC via market_order (portfolio-value-based sizing)
        if self._bitcoin in weights.index:
            btc_weight = float(weights[self._bitcoin])
            pv         = float(self._algo.portfolio.total_portfolio_value)
            if pv > 0 and btc_weight > 0:
                btc_price = float(
                    self._algo.securities[self._bitcoin].price)
                if btc_price > 0:
                    target_value  = pv * btc_weight
                    current_value = float(
                        self._algo.portfolio[self._bitcoin].holdings_value)
                    # Only reorder if drift > 1% of portfolio
                    if abs(target_value - current_value) / pv > 0.01:
                        quantity = (target_value - current_value) / btc_price
                        self._algo.market_order(self._bitcoin, quantity)

    def _liquidate(self):
        for sym in self._symbols:
            try:
                if self._algo.portfolio[sym].invested:
                    self._algo.liquidate(sym)
            except Exception:
                pass
