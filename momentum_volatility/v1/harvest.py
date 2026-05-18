# region imports
from AlgorithmImports import *
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# endregion


class HarvestSleeve:
    """
    #238 Long Short Harvest — PSR 99.44 (dominant anchor).
    Long sleeve: top-4 mega-cap stocks, VIX/ML regime-adaptive sizing.
    Short sleeve: Hurst/ATR extension trend-exhaustion candidates.
    Gross exposures scaled by PSR weight fraction W.
    """

    def __init__(self, algo, w, spy_sym, gld_sym):
        self._algo       = algo
        self._spy_sym    = spy_sym
        self._gld_sym    = gld_sym
        self.long_gross  = 0.90 * w
        self.short_gross = 0.60 * w

        self.top_set      = set()
        self._last_top_mo = -1
        self._active      = []
        self._entry       = {}
        self._long_trail  = {}

        # Long sleeve params
        self._ml_tilt    = 0.25
        self._top_wmax   = 0.35
        self._top_wmin   = 0.0
        self._lt1, self._lt2, self._lt3 = 0.095, 0.070, 0.0485

        # Short sleeve params
        self._lookback   = 260
        self._n_list     = [10, 10, 40, 60, 90, 100]
        self._sma_len    = 195
        self._ext_k      = 2.0
        self._mom_k      = 1.75
        self._score_thr  = 0.85
        self._stop_atr   = 2.0
        self._top_n      = 1

        # ML
        self._model   = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42)
        self._scaler  = StandardScaler()
        self._trained = False

        # CBOE VIX custom data
        self._vix_sym = algo.add_data(
            HarvestSleeve.CBOE_VIX, "VIX_H", Resolution.DAILY).symbol

    # ── Universe helpers ──────────────────────────────────────────────────────

    def coarse_selection(self, coarse):
        filtered = [c for c in coarse
                    if c.has_fundamental_data
                    and c.price is not None and c.price > 5
                    and c.dollar_volume is not None and c.dollar_volume > 2e7]
        filtered.sort(key=lambda c: c.dollar_volume, reverse=True)
        return [c.symbol for c in filtered[:2000]]

    def update_top_set(self, fine):
        if self._algo.time.month != self._last_top_mo:
            fine_mc = [f for f in fine if f.market_cap and f.market_cap > 0]
            fine_mc.sort(key=lambda f: f.market_cap, reverse=True)
            self.top_set = set(f.symbol for f in fine_mc[:4])
            self._last_top_mo = self._algo.time.month

    def set_active(self, syms):
        self._active = syms

    # ── ML ────────────────────────────────────────────────────────────────────

    def _get_features(self, vc, sc):
        if len(vc) < 50 or len(sc) < 200:
            return None
        cv = vc[-1]
        vm20 = np.mean(vc[-20:]); vm50 = np.mean(vc[-50:])
        vs   = np.std(vc[-20:])
        vz   = (cv - vm20) / vs if vs > 0 else 0.0
        vp   = float(np.sum(vc < cv)) / len(vc)
        sc_c = sc[-1]
        sm50 = np.mean(sc[-50:]); sm200 = np.mean(sc[-200:])
        vol  = np.std(np.diff(sc[-21:]) / sc[-21:-1])
        return [float(cv), float(vz), float(vp),
                float(cv / vm20) if vm20 else 1.0,
                float(cv / vm50) if vm50 else 1.0,
                float(sc[-1]/sc[-5]-1), float(sc[-1]/sc[-10]-1),
                float(sc[-1]/sc[-20]-1),
                float(sc_c/sm50) if sm50 else 1.0,
                float(sc_c/sm200) if sm200 else 1.0,
                float(vol * np.sqrt(252))]

    def train_model(self):
        if self._algo.is_warming_up:
            return
        vh = self._algo.history([self._vix_sym], 800, Resolution.DAILY)
        sh = self._algo.history([self._spy_sym], 800, Resolution.DAILY)
        if vh.empty or sh.empty:
            return
        try:
            vc = vh.loc[self._vix_sym]["close"].values
            sc = sh.loc[self._spy_sym]["close"].values
        except Exception:
            return
        if len(vc) < 504 or len(sc) < 504:
            return
        X, y = [], []
        for i in range(200, len(sc) - 21):
            f = self._get_features(vc[:i], sc[:i])
            if f:
                X.append(f)
                y.append(1 if sc[i+21]/sc[i] > 0.02 else 0)
        if len(X) < 100:
            return
        X = np.array(X)
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X), np.array(y))
        self._trained = True

    # ── Long sleeve ───────────────────────────────────────────────────────────

    def _safe_set(self, sym, w):
        pv = float(self._algo.portfolio.total_portfolio_value)
        if pv <= 0:
            return
        mr = float(self._algo.portfolio.margin_remaining)
        w  = float(np.clip(w, -max(0.0, mr/pv), max(0.0, mr/pv)))
        self._algo.set_holdings(sym, w)

    def _ensure_trail(self, sym, tw):
        if not self._algo.securities.contains_key(sym):
            return
        px = float(self._algo.securities[sym].price)
        if px <= 0:
            return
        st = self._long_trail.get(sym)
        if st is None:
            self._long_trail[sym] = {"high": px, "stage": 0, "target_w": float(tw)}
        else:
            st["target_w"] = float(tw)

    def _cap_renorm(self, weights, total, wmin, wmax):
        w = np.array(weights, dtype=float)
        n = len(w)
        if n == 0:
            return w
        if wmin > 0: w = np.maximum(w, wmin)
        if wmax > 0: w = np.minimum(w, wmax)
        for _ in range(10):
            diff = total - float(np.sum(w))
            if abs(diff) < 1e-8:
                break
            adj = ([i for i in range(n) if w[i] < wmax - 1e-12] if diff > 0
                   else [i for i in range(n) if w[i] > wmin + 1e-12])
            if not adj:
                break
            incr = diff / len(adj)
            for i in adj:
                w[i] += incr
            if wmin > 0: w = np.maximum(w, wmin)
            if wmax > 0: w = np.minimum(w, wmax)
        return w

    def _pick_overweight(self, syms):
        best, bc = syms[0], -1.0
        for sym in syms:
            sec = (self._algo.securities[sym]
                   if self._algo.securities.contains_key(sym) else None)
            if sec and sec.fundamentals and sec.fundamentals.market_cap:
                cap = float(sec.fundamentals.market_cap)
                if cap > bc:
                    bc = cap; best = sym
        return best

    def _allocate_top(self, total_w, ml_bullish=False):
        syms = list(self.top_set)
        if not syms:
            return
        TW = float(total_w)
        if TW <= 0:
            for sym in syms:
                if (self._algo.portfolio[sym].invested
                        and self._algo.portfolio[sym].quantity > 0):
                    self._algo.liquidate(sym)
                self._long_trail.pop(sym, None)
            return
        n = len(syms)
        weights = np.array([TW / n] * n, dtype=float)
        if ml_bullish and self._ml_tilt > 0 and n >= 2:
            ow    = self._pick_overweight(syms)
            i_ow  = syms.index(ow)
            extra = (TW / n) * self._ml_tilt
            weights[i_ow] += extra
            sub = extra / (n - 1)
            for i in range(n):
                if i != i_ow:
                    weights[i] -= sub
        weights = self._cap_renorm(weights, TW, self._top_wmin, self._top_wmax)
        for i, sym in enumerate(syms):
            w = float(weights[i])
            self._safe_set(sym, w)
            self._ensure_trail(sym, w)
        if self._algo.portfolio[self._spy_sym].invested:
            self._algo.liquidate(self._spy_sym)

    def _liquidate_non_top_longs(self):
        for kvp in self._algo.portfolio:
            sym = kvp.Key
            if sym.security_type != SecurityType.EQUITY:
                continue
            if sym in (self._spy_sym, self._gld_sym):
                continue
            h = kvp.Value
            if h.invested and h.quantity > 0 and sym not in self.top_set:
                self._algo.liquidate(sym)
                self._long_trail.pop(sym, None)

    def check_long(self):
        if self._algo.is_warming_up:
            return
        self._liquidate_non_top_longs()
        vh = self._algo.history([self._vix_sym], 100, Resolution.DAILY)
        sh = self._algo.history([self._spy_sym], 200, Resolution.DAILY)
        if vh.empty or sh.empty:
            return
        try:
            vc = vh.loc[self._vix_sym]["close"].values
            sc = sh.loc[self._spy_sym]["close"].values
        except Exception:
            return
        if len(vc) < 50 or len(sc) < 200:
            return

        cv     = float(vc[-1])
        vma    = float(np.mean(vc[-20:]))
        vp80   = float(np.percentile(vc, 80))
        sc_c   = float(sc[-1])
        sm50   = float(np.mean(sc[-50:]))
        sm200  = float(np.mean(sc[-200:]))
        r5     = float(sc[-1] / sc[-5] - 1)

        ml_bullish = False
        if self._trained:
            f = self._get_features(vc, sc)
            if f is not None:
                pa = self._model.predict_proba(self._scaler.transform([f]))[0]
                ml_bullish = float(pa[1]) > 0.6 if len(pa) == 2 else False

        LG = float(self.long_gross)
        if cv > vp80 and r5 < -0.03:
            w = 1.0 if ml_bullish else 0.85
            self._allocate_top(LG * w, ml_bullish)
            self._safe_set(self._gld_sym, LG * (1.0 - w))
        elif cv < 13 and sc_c > sm50 * 1.05:
            self._allocate_top(LG * 0.40, ml_bullish)
            self._safe_set(self._gld_sym, LG * 0.40)
        elif 20 < cv < vma:
            w = 0.85 if ml_bullish else 0.70
            self._allocate_top(LG * w, ml_bullish)
            self._safe_set(self._gld_sym, LG * (1.0 - w))
        elif cv > vma * 1.2:
            self._allocate_top(0.0, ml_bullish)
            self._safe_set(self._gld_sym, LG * 0.50)
        elif sc_c > sm200:
            base = 0.90 if ml_bullish else 0.70
            self._allocate_top(LG * base, ml_bullish)
            self._safe_set(self._gld_sym, LG * (1.0 - base))
        else:
            self._allocate_top(LG * 0.30, ml_bullish)
            self._safe_set(self._gld_sym, LG * 0.50)

    def risk_long(self):
        if self._algo.is_warming_up:
            return
        for sym in list(self._long_trail.keys()):
            p = self._algo.portfolio[sym]
            if (not self._algo.securities.contains_key(sym)
                    or not p.invested or p.quantity <= 0
                    or sym not in self.top_set):
                self._long_trail.pop(sym, None)
                continue
            px = float(self._algo.securities[sym].price)
            if px <= 0:
                continue
            st = self._long_trail[sym]
            if px > float(st["high"]):
                st["high"] = px
            high = float(st["high"])
            dd   = (high - px) / high
            stage, fw = int(st["stage"]), float(st["target_w"])
            if stage == 0 and dd >= self._lt1:
                self._safe_set(sym, fw * 2/3)
                st["stage"] = 1; st["high"] = px
            elif stage == 1 and dd >= self._lt2:
                self._safe_set(sym, fw * 1/3)
                st["stage"] = 2; st["high"] = px
            elif stage == 2 and dd >= self._lt3:
                self._algo.liquidate(sym)
                self._long_trail.pop(sym, None)

    # ── Short sleeve ──────────────────────────────────────────────────────────

    def _atr(self, df, n):
        w = df.shape[0]
        if w < n + 1:
            return None
        s = sum(max(float(df["high"].iloc[w-i]) - float(df["low"].iloc[w-i]),
                    abs(float(df["high"].iloc[w-i]) - float(df["close"].iloc[w-i])),
                    abs(float(df["low"].iloc[w-i])  - float(df["close"].iloc[w-i])))
                for i in range(1, n+1))
        return s / n

    def _hurst_like(self, df, n, bump):
        atr = self._atr(df, n)
        if atr is None or atr <= 0:
            return None
        span = float(df["high"].tail(n).max()) - float(df["low"].tail(n).min())
        if span <= 0:
            return None
        h = (np.log(span) - np.log(atr)) / np.log(float(n))
        h += bump if h > 0.45 else -bump
        return float(h)

    def _score(self, symbol):
        df = self._algo.history(symbol, self._lookback, Resolution.DAILY)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol)
        if len(df) < max(self._n_list) + 6:
            return None
        hvals = [self._hurst_like(df, n, 0.01 + 0.0002*n) for n in self._n_list]
        hvals = [h for h in hvals if h is not None]
        if len(hvals) < 4:
            return None
        havg  = sum(hvals) / len(hvals)
        agree = sum(1 for x in hvals if x > 0.6)
        c_now = float(df["close"].iloc[-1])
        sma   = float(df["close"].tail(self._sma_len).mean())
        atr20 = self._atr(df, 20)
        if atr20 is None or atr20 <= 0:
            return None
        c5 = float(df["close"].iloc[-5])
        ext_ok = (c_now - sma)  > self._ext_k * atr20
        mom_ok = (c_now - c5)   > self._mom_k * atr20
        score  = havg + 0.02 * max(0, agree - 3)
        return float(score), bool(ext_ok), bool(mom_ok), c_now, float(atr20)

    def rebalance_short(self):
        if self._algo.is_warming_up or not self._active:
            return
        scored = []
        for sym in self._active:
            if sym == self._spy_sym:
                continue
            if (not self._algo.securities.contains_key(sym)
                    or not self._algo.securities[sym].has_data):
                continue
            out = self._score(sym)
            if out is None:
                continue
            score, ext_ok, mom_ok, c_now, atr20 = out
            if score >= self._score_thr and ext_ok and mom_ok:
                scored.append((score, sym, c_now, atr20))
        scored.sort(reverse=True, key=lambda x: x[0])
        picked   = scored[:self._top_n]
        selected = {sym for _, sym, _, _ in picked}

        for kvp in self._algo.portfolio:
            sym = kvp.Key
            if sym in (self._spy_sym, self._gld_sym):
                continue
            h = kvp.Value
            if h.invested and h.quantity < 0 and sym not in selected:
                self._algo.liquidate(sym)
                self._entry.pop(sym, None)

        if selected:
            w = -abs(self.short_gross) / len(selected)
            for _, sym, c_now, atr20 in picked:
                self._safe_set(sym, w)
                if sym not in self._entry:
                    self._entry[sym] = {"entry_price": c_now, "entry_atr": atr20}

    def risk_short(self):
        if self._algo.is_warming_up:
            return
        exits = []
        for sym, info in list(self._entry.items()):
            if (not self._algo.securities.contains_key(sym)
                    or not self._algo.portfolio[sym].invested
                    or self._algo.portfolio[sym].quantity >= 0):
                self._entry.pop(sym, None)
                continue
            price = float(self._algo.securities[sym].price)
            entry = float(info["entry_price"])
            atr   = float(info["entry_atr"])
            if atr <= 0:
                continue
            if (price - entry) > self._stop_atr * atr:
                exits.append(sym)
        for sym in exits:
            self._algo.liquidate(sym)
            self._entry.pop(sym, None)

    # ── CBOE VIX custom data source ───────────────────────────────────────────

    class CBOE_VIX(PythonData):
        def get_source(self, config, date, is_live):
            return SubscriptionDataSource(
                "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                SubscriptionTransportMedium.REMOTE_FILE)

        def reader(self, config, line, date, is_live):
            if not (line.strip() and line[0].isdigit()):
                return None
            d = line.split(",")
            try:
                bar = HarvestSleeve.CBOE_VIX()
                bar.symbol   = config.symbol
                bar.time     = datetime.strptime(d[0], "%m/%d/%Y")
                bar.value    = float(d[4])
                bar["close"] = float(d[4])
                bar["open"]  = float(d[1])
                bar["high"]  = float(d[2])
                bar["low"]   = float(d[3])
                return bar
            except Exception:
                return None
