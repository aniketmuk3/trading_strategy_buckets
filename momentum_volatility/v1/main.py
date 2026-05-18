# region imports
from AlgorithmImports import *
from datetime import timedelta
from harvest import HarvestSleeve
from put_hedge import PutHedgeSleeve
from xasset import XAssetSleeve
from momvol import MomVolSleeve
from valmom import ValMomSleeve
# endregion

# =============================================================================
# BUCKET A — CONSOLIDATED LONG/SHORT MOMENTUM & VOLATILITY
# SRP 199 | May 2026
#
# PSR-weighted blend of 5 strategies (PSR total = 157.31):
#   #238 Long Short Harvest          PSR 99.44 → W = 0.632  (anchor)
#   #68  VIX-Ranked SPY Put Hedge    PSR 20.48 → W = 0.130
#   #226 Cross-Asset TS Momentum     PSR 19.56 → W = 0.124
#   #428 Monthly LS Momentum-Vol     PSR 10.51 → W = 0.067
#   #342 Large-Cap Value Momentum LS PSR  7.32 → W = 0.047
# =============================================================================

W_HARVEST   = 99.44 / 157.31   # 0.632
W_PUT_HEDGE = 20.48 / 157.31   # 0.130
W_XASSET    = 19.56 / 157.31   # 0.124
W_MOMVOL    = 10.51 / 157.31   # 0.067
W_VALMOM    =  7.32 / 157.31   # 0.047


class BucketA_Consolidated(QCAlgorithm):

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5 * 365))
        self.set_cash(1_000_000)
        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.free_portfolio_value_percentage = 0.025
        self.settings.seed_initial_prices = True

        # ── Shared instruments ────────────────────────────────────────────────
        spy_eq        = self.add_equity("SPY", Resolution.DAILY,
                                        data_normalization_mode=DataNormalizationMode.RAW)
        self._spy     = spy_eq
        self._spy_sym = spy_eq.symbol
        self.set_benchmark(self._spy_sym)
        self._gld_sym = self.add_equity("GLD", Resolution.DAILY).symbol
        self._vix     = self.add_index("VIX")

        # ── Instantiate sleeves ───────────────────────────────────────────────
        self._harvest   = HarvestSleeve(self, W_HARVEST, self._spy_sym, self._gld_sym)
        self._put_hedge = PutHedgeSleeve(self, W_PUT_HEDGE, self._spy, self._vix)
        self._xasset    = XAssetSleeve(self, W_XASSET)
        self._momvol    = MomVolSleeve(self, W_MOMVOL)
        self._valmom    = ValMomSleeve(self, W_VALMOM)

        # ── Universe (shared: harvest top-set/short, #428, #342) ──────────────
        self.universe_settings.resolution = Resolution.DAILY
        month_start = self.date_rules.month_start("SPY")
        self.universe_settings.schedule.on(month_start)
        self.add_universe(self._harvest.coarse_selection, self._fine_selection)

        # ── Schedule: #238 Harvest ────────────────────────────────────────────
        self.schedule.on(self.date_rules.every_day("SPY"),
                         self.time_rules.after_market_open("SPY", 30),
                         self._harvest.check_long)
        self.schedule.on(self.date_rules.every_day("SPY"),
                         self.time_rules.after_market_open("SPY", 90),
                         self._harvest.risk_long)
        self.schedule.on(self.date_rules.every(DayOfWeek.MONDAY),
                         self.time_rules.after_market_open("SPY", 30),
                         self._harvest.rebalance_short)
        self.schedule.on(self.date_rules.every_day("SPY"),
                         self.time_rules.after_market_open("SPY", 160),
                         self._harvest.risk_short)
        self.schedule.on(self.date_rules.month_start("SPY"),
                         self.time_rules.after_market_open("SPY", 60),
                         self._harvest.train_model)

        # ── Schedule: #68 Put Hedge ───────────────────────────────────────────
        self.schedule.on(self.date_rules.every_day("SPY"),
                         self.time_rules.after_market_open("SPY", 30),
                         self._put_hedge.daily_rebalance)

        # ── Schedule: #226 Cross-Asset ────────────────────────────────────────
        self.schedule.on(self.date_rules.month_end("SPY"),
                         self.time_rules.after_market_close("SPY", 60),
                         self._xasset.rebalance)

        # ── Schedule: #428 MomVol + #342 ValMom ──────────────────────────────
        self.schedule.on(month_start, self.time_rules.at(8, 0),
                         self._momvol.rebalance)
        self.schedule.on(month_start, self.time_rules.at(8, 0),
                         self._valmom.rebalance)

        self.set_warm_up(timedelta(420))

    def _fine_selection(self, fine):
        today = self.time.date()
        self._harvest.update_top_set(fine)
        kept = []
        for f in fine:
            if not f.market_cap or f.market_cap < 1_000_000_000:
                continue
            sr = f.security_reference
            if sr is None or sr.ipo_date is None:
                continue
            if (today - sr.ipo_date.date()).days < 365:
                continue
            if (f.security_reference.security_type != 'ST00000001'
                    or f.security_reference.is_depositary_receipt
                    or f.company_reference.is_reit
                    or f.asset_classification.morningstar_sector_code
                       == MorningstarSectorCode.FINANCIAL_SERVICES):
                continue
            kept.append(f)
        kept.sort(key=lambda f: f.dollar_volume, reverse=True)
        universe_syms = [f.symbol for f in kept[:150]]
        self._harvest.set_active(universe_syms)
        return list(set(universe_syms) | self._harvest.top_set)

    def on_securities_changed(self, changes):
        self._momvol.on_securities_changed(changes)
        self._valmom.on_securities_changed(changes)

    def on_warmup_finished(self):
        self._harvest.train_model()
        self._xasset.rebalance()
        self._momvol.rebalance()
        self._valmom.rebalance()
        self._put_hedge.daily_rebalance()
