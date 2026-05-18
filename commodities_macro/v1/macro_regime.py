# region imports
from AlgorithmImports import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
# endregion

# =============================================================================
# MacroRegimeAllocator — Bucket C
#
# Classifies the macro environment into 4 regimes using a Gradient
# Boosting classifier trained on yield curve, inflation proxy,
# equity trend, and volatility features pulled from FRED + QC data.
#
# Regime classes:
#   0 — Risk-on growth       → macro rotation + momentum dominate
#   1 — Reflation/inflation  → commodities + real assets dominate
#   2 — Risk-off defensive   → all-weather ballast + reduced risk-on
#   3 — Crisis/deleveraging  → all-weather + cash; futures gated off
#
# Fix: REGIME_ALLOC rows converted to plain lists via .tolist() before
# storing in self._allocs to avoid numpy array type issues downstream.
# =============================================================================

REGIME_ALLOC = np.array([
    # macro_rot  mom_rot  futures  all_weather
    [0.55,      0.25,    0.12,    0.08],   # 0: risk-on growth
    [0.40,      0.20,    0.25,    0.15],   # 1: reflation
    [0.25,      0.20,    0.05,    0.50],   # 2: risk-off defensive
    [0.10,      0.10,    0.00,    0.80],   # 3: crisis
], dtype=float)

_KEYS = ["macro_rot", "mom_rot", "futures", "all_weather"]


class MacroRegimeAllocator:

    def __init__(self, algo, spy_sym, vix_security):
        self._algo    = algo
        self._spy_sym = spy_sym
        self._vix     = vix_security
        self._model   = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, random_state=42)
        self._scaler  = StandardScaler()
        self._trained = False
        # Default: risk-off defensive until model trains
        # Use .tolist() to ensure plain Python floats, not numpy types
        self._allocs  = dict(zip(_KEYS, REGIME_ALLOC[2].tolist()))

        # FRED macro factors
        self._factors = [
            algo.add_data(Fred, t, Resolution.DAILY).symbol
            for t in ["T10Y3M", "DFF", "VIXCLS"]
        ]

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self, vix_arr, spy_arr, factor_df):
        if len(vix_arr) < 200 or len(spy_arr) < 200:
            return None

        vix_now  = float(vix_arr[-1])
        vix_sma  = float(np.mean(vix_arr[-60:]))
        vix_rank = float((vix_now - np.min(vix_arr[-252:])) / (
            np.max(vix_arr[-252:]) - np.min(vix_arr[-252:]) + 1e-8))
        vix_mom  = float(vix_arr[-1] / vix_arr[-21] - 1) if len(vix_arr) > 21 else 0.0

        spy_sma50    = float(np.mean(spy_arr[-50:]))
        spy_sma200   = float(np.mean(spy_arr[-200:]))
        spy_ret1m    = float(spy_arr[-1] / spy_arr[-21]  - 1) if len(spy_arr) > 21  else 0.0
        spy_ret3m    = float(spy_arr[-1] / spy_arr[-63]  - 1) if len(spy_arr) > 63  else 0.0
        spy_ret6m    = float(spy_arr[-1] / spy_arr[-126] - 1) if len(spy_arr) > 126 else 0.0
        spy_above200 = float(spy_arr[-1] > spy_sma200)
        spy_above50  = float(spy_arr[-1] > spy_sma50)

        spy_rets     = np.diff(spy_arr[-22:]) / spy_arr[-22:-1]
        realized_vol = float(np.std(spy_rets) * np.sqrt(252)) if len(spy_rets) > 1 else 0.2

        yc_slope, dff_level = 0.0, 0.0
        if factor_df is not None and not factor_df.empty:
            if "T10Y3M" in factor_df.columns:
                yc_s = factor_df["T10Y3M"].dropna()
                if len(yc_s) > 0:
                    yc_slope = float(yc_s.iloc[-1])
            if "DFF" in factor_df.columns:
                dff_s = factor_df["DFF"].dropna()
                if len(dff_s) > 0:
                    dff_level = float(dff_s.iloc[-1])

        return [
            vix_now, vix_rank, vix_sma, vix_mom,
            spy_ret1m, spy_ret3m, spy_ret6m,
            spy_above50, spy_above200, realized_vol,
            yc_slope, dff_level,
        ]

    def _label_regime(self, vix_val, vix_rank, yc_slope, spy_above200, spy_ret3m):
        if vix_val > 30 or vix_rank > 0.80:
            return 3   # crisis
        if vix_rank > 0.55 or not spy_above200:
            return 2   # risk-off
        if yc_slope > 0.5 and spy_ret3m > 0.03:
            return 0   # risk-on growth
        if yc_slope < 0 or vix_rank < 0.25:
            return 1   # reflation
        return 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        if self._algo.is_warming_up:
            return
        try:
            vh  = self._algo.history([self._vix.symbol], 800, Resolution.DAILY)
            sh  = self._algo.history([self._spy_sym],    800, Resolution.DAILY)
            fh  = self._algo.history(self._factors,      800, Resolution.DAILY)
            if vh.empty or sh.empty:
                return
            vix_arr = vh.loc[self._vix.symbol]["close"].values
            spy_arr = sh.loc[self._spy_sym]["close"].values
            fdf = None
            if not fh.empty:
                fdf = fh["value"].unstack(0)
                fdf.columns = [str(c).split()[0] for c in fdf.columns]
        except Exception:
            return

        if len(vix_arr) < 250 or len(spy_arr) < 250:
            return

        X, y = [], []
        for i in range(200, len(vix_arr)):
            vc = vix_arr[:i]
            sc = spy_arr[:i] if i <= len(spy_arr) else spy_arr
            fd = fdf.iloc[:i] if fdf is not None else None
            feats = self._build_features(vc, sc, fd)
            if feats is None:
                continue
            vr  = float((vc[-1] - np.min(vc[-252:])) / (
                np.max(vc[-252:]) - np.min(vc[-252:]) + 1e-8))
            yc  = 0.0
            if fd is not None and "T10Y3M" in fd.columns:
                s = fd["T10Y3M"].dropna()
                if len(s) > 0:
                    yc = float(s.iloc[-1])
            r3  = float(sc[-1] / sc[-63] - 1) if len(sc) > 63 else 0.0
            ab200 = float(sc[-1] > np.mean(sc[-200:]))
            X.append(feats)
            y.append(self._label_regime(float(vc[-1]), vr, yc, ab200, r3))

        if len(X) < 50:
            return
        X_arr = np.array(X)
        self._scaler.fit(X_arr)
        self._model.fit(self._scaler.transform(X_arr), np.array(y))
        self._trained = True
        self._refresh(vix_arr, spy_arr, fdf)

    def _refresh(self, vix_arr, spy_arr, fdf):
        feats = self._build_features(vix_arr, spy_arr, fdf)
        if feats is None or not self._trained:
            return
        try:
            probs      = self._model.predict_proba(
                self._scaler.transform([feats]))[0]
            full_probs = np.zeros(4)
            for i, cls in enumerate(self._model.classes_):
                full_probs[int(cls)] = probs[i]
            alloc_vec  = full_probs @ REGIME_ALLOC
            alloc_vec  = alloc_vec / (alloc_vec.sum() + 1e-8)
            # Use .tolist() — ensures plain Python floats, not numpy scalars
            self._allocs = dict(zip(_KEYS, alloc_vec.tolist()))
        except Exception:
            pass

    def get_allocations(self):
        if not self._trained:
            return self._allocs
        try:
            vh  = self._algo.history([self._vix.symbol], 600, Resolution.DAILY)
            sh  = self._algo.history([self._spy_sym],    600, Resolution.DAILY)
            fh  = self._algo.history(self._factors,      600, Resolution.DAILY)
            va  = vh.loc[self._vix.symbol]["close"].values if not vh.empty else np.array([])
            sa  = sh.loc[self._spy_sym]["close"].values    if not sh.empty else np.array([])
            fdf = None
            if not fh.empty:
                fdf = fh["value"].unstack(0)
                fdf.columns = [str(c).split()[0] for c in fdf.columns]
            self._refresh(va, sa, fdf)
        except Exception:
            pass
        return self._allocs
