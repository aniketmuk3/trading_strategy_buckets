# region imports
from AlgorithmImports import *
from datetime import timedelta
from regime import RegimeAllocator
from put_overlay import PutOverlaySleeve
from put_harvest import PutHarvestSleeve
from straddle import StradleSleeve
from vrp_gate import VRPGateSleeve
from vix_timing import VIXTimingSleeve
# endregion

# =============================================================================
# BUCKET B — CONSOLIDATED VOLATILITY & OPTIONS
# SRP 199 | May 2026
#
# Dynamic regime-based allocation across 5 volatility strategies.
# A Random Forest regime classifier reads VIX structure, VRP, and
# momentum features monthly and outputs capital allocation weights
# across sleeves. Hard risk rules override ML in crisis conditions.
#
# Strategies:
#   #228 Long SPY VIX-Ranked Put Overlay      (defensive/stress regime)
#   #271 Systematic Short SPY Put Harvest     (calm/VRP-positive regime)
#   #99  VIX Z-Score Straddle Mean Reversion  (extreme z-score regimes)
#   #70  SPX Put VRP Harvest Regime-Gated     (low/mod IV + strike density)
#   #49  VIX Predicts Stock Index Returns     (extreme VIX percentile tilt)
#
# Hard risk overrides:
#   - VIX > 40: liquidate all short-vol positions immediately
#   - #271 never sells when VIX rank > 0.6
#   - #99 straddle size capped at 2 contracts
#
# Fixes applied:
#   - Replaced every_day("SPX") date rule with every_day(spy_sym)
#   - VRP rebalance now wrapped in _vrp_rebalance() to pass alloc weight
#   - Crisis liquidate calls use symbol-safe liquidate()
# =============================================================================

VIX_CRISIS_LEVEL = 40.0
VIX_HARVEST_GATE = 0.60


class BucketB_Consolidated(QCAlgorithm):

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5 * 365))
        self.set_cash(1_000_000)
        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.seed_initial_prices = True
        self.settings.automatic_indicator_warm_up = True
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # ── Shared instruments ────────────────────────────────────────────────
        self._spy     = self.add_equity(
            "SPY", data_normalization_mode=DataNormalizationMode.RAW)
        self._spy_sym = self._spy.symbol
        self.set_benchmark(self._spy_sym)
        self._vix     = self.add_index("VIX")
        self._oef_sym = self.add_equity(
            "OEF", data_normalization_mode=DataNormalizationMode.RAW).symbol

        # ── Regime allocator ──────────────────────────────────────────────────
        self._regime = RegimeAllocator(self, self._spy_sym, self._vix)

        # ── Sleeves ───────────────────────────────────────────────────────────
        self._overlay  = PutOverlaySleeve(self, self._spy, self._vix)
        self._harvest  = PutHarvestSleeve(self, self._spy)
        self._straddle = StradleSleeve(self, self._vix)
        self._vrp      = VRPGateSleeve(self)
        self._timing   = VIXTimingSleeve(self, self._vix, self._oef_sym)

        # ── Schedules ─────────────────────────────────────────────────────────
        # Monthly retrain
        self.schedule.on(
            self.date_rules.month_start(self._spy_sym),
            self.time_rules.after_market_open(self._spy_sym, 60),
            self._retrain)

        # Daily: overlay + harvest + straddle
        self.schedule.on(
            self.date_rules.every_day(self._spy_sym),
            self.time_rules.after_market_open(self._spy_sym, 30),
            self._daily_rebalance)

        # Daily: VRP gate — wrapped so alloc weight can be passed
        # Uses spy_sym as anchor (SPX is an index, not valid for date rules)
        self.schedule.on(
            self.date_rules.every_day(self._spy_sym),
            self.time_rules.after_market_open(self._spy_sym, 1),
            self._vrp_rebalance)

        # Weekly: VIX timing
        self.schedule.on(
            self.date_rules.week_start(self._spy_sym),
            self.time_rules.after_market_open(self._spy_sym, 30),
            self._timing_rebalance)

        self.set_warm_up(timedelta(550))

    # ── Crisis override ───────────────────────────────────────────────────────

    def _crisis_check(self):
        """Hard rule: liquidate all short-vol if VIX spikes above crisis level."""
        if float(self._vix.price) > VIX_CRISIS_LEVEL:
            self._harvest.liquidate_all()
            self._straddle.liquidate_short()
            self._vrp.liquidate_all()
            return True
        return False

    # ── Scheduled handlers ────────────────────────────────────────────────────

    def _retrain(self):
        if self.is_warming_up:
            return
        self._regime.train()

    def _daily_rebalance(self):
        if self.is_warming_up:
            return
        in_crisis = self._crisis_check()
        allocs    = self._regime.get_allocations()

        # #228 overlay always runs
        self._overlay.daily_rebalance(allocs["overlay"])

        # #271 harvest gated by VIX rank and crisis
        vix_rank = self._overlay.vix_rank
        if not in_crisis and vix_rank <= VIX_HARVEST_GATE:
            self._harvest.sell_puts(allocs["harvest"])
        else:
            self._harvest.liquidate_all()

        # #99 straddle
        self._straddle.rebalance(allocs["straddle"])

    def _vrp_rebalance(self):
        """Wrapper so alloc weight can be fetched and passed to VRP sleeve."""
        if self.is_warming_up:
            return
        allocs = self._regime.get_allocations()
        self._vrp.rebalance(allocs["vrp"])

    def _timing_rebalance(self):
        if self.is_warming_up:
            return
        allocs = self._regime.get_allocations()
        self._timing.rebalance(allocs["timing"])

    # ── Order event routing ───────────────────────────────────────────────────

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED and order_event.is_assignment:
            self._harvest.on_assignment(order_event)
            self._straddle.on_assignment(order_event)

    def on_warmup_finished(self):
        self._regime.train()
        allocs = self._regime.get_allocations()
        self._overlay.daily_rebalance(allocs["overlay"])
        self._timing.rebalance(allocs["timing"])
