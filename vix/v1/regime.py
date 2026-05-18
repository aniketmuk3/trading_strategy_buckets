# region imports
from AlgorithmImports import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# endregion

# =============================================================================
# RegimeAllocator — Bucket B
#
# Classifies the current volatility regime using a Random Forest trained
# on rolling historical VIX and SPY features. Maps regime probabilities
# to smooth capital allocation weights across the 5 Bucket B sleeves.
#
# Regime classes:
#   0 — Low-vol complacent    → harvest + VRP dominate
#   1 — Low-vol trending      → harvest + momentum tilt
#   2 — Mean-reversion        → straddle + overlay moderate
#   3 — Vol spike             → overlay + straddle dominate
#   4 — High-vol stress       → overlay dominates, short-vol off
# =============================================================================

REGIME_ALLOC = np.array([
    # overlay  harvest  straddle  vrp   timing
    [0.10,    0.45,    0.20,    0.20,  0.05],  # 0: low-vol complacent
    [0.15,    0.40,    0.15,    0.20,  0.10],  # 1: low-vol trending
    [0.25,    0.20,    0.35,    0.10,  0.10],  # 2: mean-reversion
    [0.50,    0.05,    0.30,    0.05,  0.10],  # 3: vol spike
    [0.75,    0.00,    0.15,    0.00,  0.10],  # 4: high-vol stress
], dtype=float)

_KEYS = ["overlay", "harvest", "straddle", "vrp", "timing"]


class RegimeAllocator:

    def __init__(self, algo, spy_sym, vix_security):
        self._algo    = algo
        self._spy_sym = spy_sym
        self._vix     = vix_security
        self._model   = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=10,
            random_state=42, class_weight="balanced")
        self._scaler  = StandardScaler()
        self._trained = False
        # Default: mean-reversion regime until model is trained
        self._allocs  = dict(zip(_KEYS, REGIME_ALLOC[2].tolist()))

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self, vix_arr, spy_arr):
        if len(vix_arr) < 252 or len(spy_arr) < 252:
            return None
        v, s = vix_arr, spy_arr

        vix_now    = float(v[-1])
        vix_1y_lo  = float(np.min(v[-252:]))
        vix_1y_hi  = float(np.max(v[-252:]))
        vix_rank   = (vix_now - vix_1y_lo) / (vix_1y_hi - vix_1y_lo + 1e-8)
        vix_2y_pct = (float(np.sum(v[-504:] < vix_now)) / 504.0
                      if len(v) >= 504 else vix_rank)

        vix_sma20  = float(np.mean(v[-20:]))
        vix_std20  = float(np.std(v[-20:]))
        vix_z20    = (vix_now - vix_sma20) / (vix_std20 + 1e-8)
        vix_sma504 = float(np.mean(v[-504:])) if len(v) >= 504 else float(np.mean(v))
        vix_std504 = float(np.std(v[-504:]))  if len(v) >= 504 else float(np.std(v))
        vix_z504   = (vix_now - vix_sma504) / (vix_std504 + 1e-8)

        vix_mom5  = float(v[-1] / v[-6]  - 1) if len(v) > 6  else 0.0
        vix_mom10 = float(v[-1] / v[-11] - 1) if len(v) > 11 else 0.0
        vix_mom20 = float(v[-1] / v[-21] - 1) if len(v) > 21 else 0.0

        spy_rets   = np.diff(s[-22:]) / s[-22:-1]
        realized20 = float(np.std(spy_rets) * np.sqrt(252)) if len(spy_rets) >= 20 else 0.2
        vrp        = vix_now / 100.0 - realized20

        spy_sma50    = float(np.mean(s[-50:]))
        spy_sma200   = float(np.mean(s[-200:]))
        spy_above50  = float(s[-1] > spy_sma50)
        spy_above200 = float(s[-1] > spy_sma200)
        spy_ret20    = float(s[-1] / s[-21] - 1) if len(s) > 21 else 0.0

        if len(v) >= 10:
            vix_d = np.diff(v[-11:])
            vix_autocorr = (float(np.corrcoef(vix_d[:-1], vix_d[1:])[0, 1])
                            if len(vix_d) > 2 else 0.0)
        else:
            vix_autocorr = 0.0

        return [
            vix_now, vix_rank, vix_2y_pct,
            vix_z20, vix_z504,
            vix_mom5, vix_mom10, vix_mom20,
            vrp, realized20,
            spy_above50, spy_above200, spy_ret20,
            vix_autocorr,
        ]

    def _label_regime(self, vix_val, vix_rank, vix_z, vrp):
        if vix_val > 30 or vix_rank > 0.85:
            return 4
        if vix_val > 22 or vix_rank > 0.65:
            return 3
        if abs(vix_z) > 1.5:
            return 2
        if vix_rank < 0.25 and vrp > 0.02:
            return 0
        return 1

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        if self._algo.is_warming_up:
            return
        try:
            vh = self._algo.history([self._vix.symbol], 800, Resolution.DAILY)
            sh = self._algo.history([self._spy_sym],   800, Resolution.DAILY)
            if vh.empty or sh.empty:
                return
            vix_arr = vh.loc[self._vix.symbol]["close"].values
            spy_arr = sh.loc[self._spy_sym]["close"].values
        except Exception:
            return
        if len(vix_arr) < 300 or len(spy_arr) < 300:
            return

        X, y = [], []
        for i in range(252, len(vix_arr)):
            vc = vix_arr[:i]
            sc = spy_arr[:i] if i <= len(spy_arr) else spy_arr
            feats = self._build_features(vc, sc)
            if feats is None:
                continue
            vr  = (vc[-1] - np.min(vc[-252:])) / (
                   np.max(vc[-252:]) - np.min(vc[-252:]) + 1e-8)
            vz  = (vc[-1] - np.mean(vc[-20:])) / (np.std(vc[-20:]) + 1e-8)
            sr  = np.diff(sc[-22:]) / sc[-22:-1] if len(sc) >= 22 else np.array([0.0])
            vrp = vc[-1] / 100.0 - float(np.std(sr) * np.sqrt(252))
            X.append(feats)
            y.append(self._label_regime(float(vc[-1]), float(vr), float(vz), float(vrp)))

        if len(X) < 50:
            return
        X_arr = np.array(X)
        self._scaler.fit(X_arr)
        self._model.fit(self._scaler.transform(X_arr), np.array(y))
        self._trained = True
        self._update_allocs(vix_arr, spy_arr)

    # ── Allocation ────────────────────────────────────────────────────────────

    def _update_allocs(self, vix_arr, spy_arr):
        feats = self._build_features(vix_arr, spy_arr)
        if feats is None or not self._trained:
            return
        probs      = self._model.predict_proba(
            self._scaler.transform([feats]))[0]
        full_probs = np.zeros(5)
        for i, cls in enumerate(self._model.classes_):
            full_probs[int(cls)] = probs[i]
        alloc_vec = full_probs @ REGIME_ALLOC
        alloc_vec = alloc_vec / (alloc_vec.sum() + 1e-8)
        self._allocs = dict(zip(_KEYS, alloc_vec.tolist()))

    def get_allocations(self):
        if not self._trained:
            return dict(zip(_KEYS, REGIME_ALLOC[2].tolist()))
        try:
            vh = self._algo.history([self._vix.symbol], 600, Resolution.DAILY)
            sh = self._algo.history([self._spy_sym],   600, Resolution.DAILY)
            if not vh.empty and not sh.empty:
                self._update_allocs(
                    vh.loc[self._vix.symbol]["close"].values,
                    sh.loc[self._spy_sym]["close"].values)
        except Exception:
            pass
        return self._allocs
