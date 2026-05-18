# region imports
from AlgorithmImports import *
from datetime import timedelta
from macro_regime import MacroRegimeAllocator
from macro_rotation import MacroRotationSleeve
from momentum_rotation import MomentumRotationSleeve
from futures_parity import FuturesParitySleeve
from all_weather import AllWeatherSleeve
# endregion

# =============================================================================
# BUCKET C — CONSOLIDATED COMMODITIES & MACRO FUTURES
# SRP 199 | May 2026
#
# Dynamic macro-regime allocation across 4 strategy sleeves.
# A Gradient Boosting regime classifier reads yield curve, inflation,
# equity trend, and commodity momentum features monthly to output
# capital allocation weights. Hard risk rules gate crisis exposure.
#
# Sleeves:
#   #72  Monthly Macro Factor Cross-Asset Rotation  (ML anchor, ~50-70%)
#   #276 Cross-Asset Momentum Rotation Vol-Scaled   (trend following, ~15-25%)
#   #233 Low-Vol Futures Risk Parity                (calm regime only, ~0-15%)
#   #146 Levered Risk-Balanced Macro Diversifier    (structural ballast, ~5-15%)
#
# Note: #26 Term Structure signal folded into FuturesParitySleeve
#       as a contract selection filter, not a standalone allocation.
#
# Hard risk overrides:
#   - #233 fully gated: only runs when VIX < 1Y median AND SPY > 200d SMA
#   - #233 gross exposure capped at 50% of its sleeve allocation
#   - Crisis (VIX > 35): reduce all risk-on positions by 50%
#
# Fixes applied:
#   - week_start() anchored to spy_sym
#   - Annual all-weather rebalance wrapped in _annual_rebalance()
#     so alloc_weight can be passed (direct schedule ref would crash)
# =============================================================================

VIX_CRISIS_LEVEL = 35.0


class BucketC_Consolidated(QCAlgorithm):

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5 * 365))
        self.set_cash(1_000_000)
        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.seed_initial_prices = True
        self.settings.daily_precise_end_time = False

        # ── Shared instruments ────────────────────────────────────────────────
        self._spy     = self.add_equity(
            "SPY", data_normalization_mode=DataNormalizationMode.RAW)
        self._spy_sym = self._spy.symbol
        self.set_benchmark(self._spy_sym)
        self._vix = self.add_index("VIX")

        # ── Regime allocator ──────────────────────────────────────────────────
        self._regime = MacroRegimeAllocator(self, self._spy_sym, self._vix)

        # ── Sleeves ───────────────────────────────────────────────────────────
        self._macro_rot   = MacroRotationSleeve(self)
        self._mom_rot     = MomentumRotationSleeve(self)
        self._futures     = FuturesParitySleeve(self, self._spy_sym, self._vix)
        self._all_weather = AllWeatherSleeve(self)

        # ── Schedules ─────────────────────────────────────────────────────────
        # Monthly: retrain + macro rotation + all-weather
        self.schedule.on(
            self.date_rules.month_start(self._spy_sym),
            self.time_rules.after_market_open(self._spy_sym, 60),
            self._monthly_rebalance)

        # Monthly: momentum rotation (earlier in day for fresh prices)
        self.schedule.on(
            self.date_rules.month_start(self._spy_sym),
            self.time_rules.at(8, 0),
            self._mom_rebalance)

        # Weekly: futures parity (fresh vol estimates needed)
        # Anchored to spy_sym — week_start() with no anchor is unreliable
        self.schedule.on(
            self.date_rules.week_start(self._spy_sym),
            self.time_rules.at(10, 0),
            self._futures_rebalance)

        # Annual: all-weather rebalance via wrapper so alloc_weight is passed
        self.schedule.on(
            self.date_rules.year_start(self._spy_sym),
            self.time_rules.midnight,
            self._annual_rebalance)

        self.set_warm_up(timedelta(420))

    # ── Crisis check ──────────────────────────────────────────────────────────

    def _in_crisis(self):
        return float(self._vix.price) > VIX_CRISIS_LEVEL

    # ── Scheduled handlers ────────────────────────────────────────────────────

    def _monthly_rebalance(self):
        if self.is_warming_up:
            return
        self._regime.train()
        allocs = self._regime.get_allocations()
        scale  = 0.5 if self._in_crisis() else 1.0
        self._macro_rot.rebalance(allocs["macro_rot"] * scale)
        self._all_weather.rebalance(allocs["all_weather"])

    def _mom_rebalance(self):
        if self.is_warming_up:
            return
        allocs = self._regime.get_allocations()
        scale  = 0.5 if self._in_crisis() else 1.0
        self._mom_rot.rebalance(allocs["mom_rot"] * scale)

    def _futures_rebalance(self):
        if self.is_warming_up:
            return
        allocs    = self._regime.get_allocations()
        gate_ok   = self._futures.vix_gate_ok()
        if gate_ok and not self._in_crisis():
            self._futures.rebalance(allocs["futures"])
        else:
            self._futures.liquidate_all()

    def _annual_rebalance(self):
        """Wrapper for annual all-weather rebalance so alloc_weight is passed."""
        if self.is_warming_up:
            return
        allocs = self._regime.get_allocations()
        self._all_weather.rebalance(allocs["all_weather"])

    def on_warmup_finished(self):
        self._regime.train()
        allocs = self._regime.get_allocations()
        self._macro_rot.rebalance(allocs["macro_rot"])
        self._mom_rot.rebalance(allocs["mom_rot"])
        self._all_weather.rebalance(allocs["all_weather"])
