#%%
# ==========================================
# Imports
# ==========================================

from binance.client import Client
from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


#%%
# ==========================================
# دریافت داده‌ی تاریخی از بایننس
# ==========================================

def get_historical_data(symbol, interval, start_time, end_time):

    client = Client()

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=getattr(Client, 'KLINE_INTERVAL_' + interval),
        start_str=start_time,
        end_str=end_time
    )

    data = pd.DataFrame(
        np.array(klines)[:, :6],
        columns=[
            'Timestamp',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]
    ).apply(pd.to_numeric)

    data.index = pd.to_datetime(data.Timestamp, unit='ms')
    data.drop(columns=['Timestamp'], inplace=True)

    return data


#%%
# بازه‌ی کلی داده رو بگیر - هرچقدر بازه بزرگ‌تر باشه پنجره‌های بیشتری برای
# walk-forward خواهی داشت
df = get_historical_data('BTCUSDT', '1HOUR', '2023-01-01', '2026-03-01').iloc[:-1]
df


#%%
# ==========================================
# ساخت پنجره‌های walk-forward
# هر پنجره: ۴ ماه train + ۲ ماه test
# پنجره‌ی بعدی با گام ۲ ماه (طول test) به جلو حرکت می‌کند
# ==========================================

def build_walk_forward_windows(data, train_months=4, test_months=2, step_months=2):

    windows = []

    start = data.index[0]
    end = data.index[-1]

    while True:

        train_start = start
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        start = start + pd.DateOffset(months=step_months)

    return windows


windows = build_walk_forward_windows(df, train_months=4, test_months=2, step_months=2)

print(f"تعداد پنجره‌های walk-forward: {len(windows)}\n")

for i, w in enumerate(windows):
    print(
        f"Window {i+1}: "
        f"Train [{w['train_start'].date()} -> {w['train_end'].date()}]  "
        f"Test [{w['test_start'].date()} -> {w['test_end'].date()}]"
    )


#%%
# ==========================================
# استراتژی
# ==========================================

class AdvancedShadowStrategy(Strategy):

    # ---------- Parameters ----------
    n_swing = 3

    tolerance = 0.002

    rr_ratio = 2.0

    r_threshold = 2.0

    max_zones = 30

    min_zone_touches = 2

    # --------------------------------

    def init(self):

        self.swing_high_index = set()
        self.swing_low_index = set()

        # Zones
        self.res_zones = []
        self.sup_zones = []

        # Variables
        self.market_trend = None
        self.last_swing_high = None
        self.last_swing_low = None
        self.last_structure = None
        self.swing_highs = []
        self.swing_lows = []

        # جلوگیری از ورود تکراری
        self.last_trade_zone = None

        # ثبت تمام برخوردهای قیمت با نواحی حمایت/مقاومت (per-instance، نه global)
        self.touch_log = []

    # ==========================================
    def update_zones(self, new_price, zones, index):

        merged = False

        for zone in zones:

            if abs(new_price - zone["level"]) / zone["level"] <= self.tolerance:

                zone["prices"].append(new_price)
                zone["level"] = np.mean(zone["prices"])
                zone["low"] = min(zone["prices"])
                zone["high"] = max(zone["prices"])
                zone["touches"] += 1
                zone["last_touch"] = index
                merged = True
                break

        if not merged:
            zones.append(
                {
                    "prices": [new_price],
                    "level": new_price,
                    "low": new_price,
                    "high": new_price,
                    "touches": 1,
                    "last_touch": index,
                }
            )

        if len(zones) > self.max_zones:
            zones.pop(0)

    # ==========================================
    def zone_available(self, level):

        if self.last_trade_zone is None:
            return True

        distance = abs(level - self.last_trade_zone) / level

        return distance > self.tolerance

    # ==========================================
    def log_zone_touch(self, zone, zone_type, high, low, close, signal, trade_taken):

        self.touch_log.append(
            {
                "time": self.data.index[-1],
                "zone_type": zone_type,
                "zone_level": zone["level"],
                "zone_low": zone["low"],
                "zone_high": zone["high"],
                "zone_touches": zone["touches"],
                "candle_high": high,
                "candle_low": low,
                "candle_close": close,
                "signal_valid": signal,
                "trade_taken": trade_taken,
            }
        )

    # ==========================================

    def next(self):

        idx = len(self.data) - 1
        n = self.n_swing

        if idx < 2 * n + 5:
            return

        # ======================================
        # Swing Detection
        # ======================================

        window_h = self.data.High[-(2 * n + 1):]
        window_l = self.data.Low[-(2 * n + 1):]

        candidate_h = window_h[n]
        candidate_l = window_l[n]

        swing_index = idx - n

        is_high = True
        is_low = True

        for i in range(2 * n + 1):

            if i == n:
                continue

            if window_h[i] >= candidate_h:
                is_high = False

            if window_l[i] <= candidate_l:
                is_low = False

        if is_high and swing_index not in self.swing_high_index:

            self.swing_high_index.add(swing_index)
            self.update_zones(candidate_h, self.res_zones, swing_index)
            self.swing_highs.append({"index": swing_index, "price": candidate_h})

        if is_low and swing_index not in self.swing_low_index:

            self.swing_low_index.add(swing_index)
            self.update_zones(candidate_l, self.sup_zones, swing_index)
            self.swing_lows.append({"index": swing_index, "price": candidate_l})

        # ======================================
        # Candle Variables
        # ======================================

        close = self.data.Close[-1]
        open_ = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        body = abs(close - open_)

        if body == 0:
            return

        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low

        r_upper = upper_shadow / body
        r_lower = lower_shadow / body

        bullish = close > open_
        bearish = close < open_

        # ======================================
        # ثبت برخوردهای قیمت با نواحی مقاومت و حمایت
        # ======================================

        for zone in self.res_zones:
            if high >= zone["low"] and high <= zone["high"]:
                self.log_zone_touch(
                    zone,
                    "resistance",
                    high,
                    low,
                    close,
                    signal=(r_upper >= self.r_threshold and close < zone["level"]),
                    trade_taken=False,
                )

        for zone in self.sup_zones:
            if low <= zone["high"] and low >= zone["low"]:
                self.log_zone_touch(
                    zone,
                    "support",
                    high,
                    low,
                    close,
                    signal=(r_lower >= self.r_threshold and close > zone["level"]),
                    trade_taken=False,
                )

        # اگر معامله باز داریم، معامله جدید باز نکن
        if self.position:
            return

        # ======================================
        # BOS
        # ======================================

        bullish_bos = False
        bearish_bos = False

        if len(self.swing_highs) >= 2:
            reference_high = self.swing_highs[-2]["price"]
            bullish_bos = close > reference_high

        if len(self.swing_lows) >= 2:
            reference_low = self.swing_lows[-2]["price"]
            bullish_bos = close > reference_low

        # ======================================
        # SHORT
        # ======================================

        if bearish_bos and r_upper >= self.r_threshold:

            for zone in self.res_zones:

                if zone["touches"] < self.min_zone_touches:
                    continue

                z_low = zone["low"]
                z_high = zone["high"]
                z_mid = zone["level"]

                if not self.zone_available(z_mid):
                    continue

                if high >= z_low and high <= z_high and close < z_mid:

                    sl = z_high
                    risk = sl - close

                    if risk <= 0:
                        continue

                    tp = close - risk * self.rr_ratio

                    self.sell(sl=sl, tp=tp)

                    self.last_trade_zone = z_mid

                    for rec in reversed(self.touch_log):
                        if (
                            rec["time"] == self.data.index[-1]
                            and rec["zone_type"] == "resistance"
                            and rec["zone_level"] == z_mid
                        ):
                            rec["trade_taken"] = True
                            break

                    return

        # ======================================
        # LONG
        # ======================================

        if bullish_bos and r_lower >= self.r_threshold:

            for zone in self.sup_zones:

                if zone["touches"] < self.min_zone_touches:
                    continue

                z_low = zone["low"]
                z_high = zone["high"]
                z_mid = zone["level"]

                if not self.zone_available(z_mid):
                    continue

                if low <= z_high and low >= z_low and close > z_mid:

                    sl = z_low
                    risk = close - sl

                    if risk <= 0:
                        continue

                    tp = close + risk * self.rr_ratio

                    self.buy(sl=sl, tp=tp)

                    self.last_trade_zone = z_mid

                    for rec in reversed(self.touch_log):
                        if (
                            rec["time"] == self.data.index[-1]
                            and rec["zone_type"] == "support"
                            and rec["zone_level"] == z_mid
                        ):
                            rec["trade_taken"] = True
                            break

                    return


#%%
# ==========================================
# اجرای walk-forward: برای هر پنجره، اپتیمایز روی train و اجرا روی test
# ==========================================

OPT_GRID = dict(
    n_swing=[3, 5, 7],
    rr_ratio=[1.5, 2, 2.5],
    r_threshold=[1.5, 2, 2.5],
)

summary_rows = []
all_test_touches = []
all_test_trades = []

for i, w in enumerate(windows):

    window_id = i + 1

    train_df = df[w["train_start"]: w["train_end"]]
    test_df = df[w["test_start"]: w["test_end"]]

    if len(train_df) < 50 or len(test_df) < 20:
        print(f"Window {window_id}: داده ناکافی، رد شد.")
        continue

    print(f"\n========== Window {window_id} ==========")
    print(f"Train: {w['train_start'].date()} -> {w['train_end'].date()} ({len(train_df)} rows)")
    print(f"Test:  {w['test_start'].date()} -> {w['test_end'].date()} ({len(test_df)} rows)")

    # ---------- اپتیمایز  train ----------
    bt_train = Backtest(
        train_df,
        AdvancedShadowStrategy,
        cash=1_000_000,
        commission=0.0001,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats_train = bt_train.optimize(
        maximize="Return [%]",
        max_tries=100,
        **OPT_GRID,
    )

    best_n_swing = stats_train._strategy.n_swing
    best_rr_ratio = stats_train._strategy.rr_ratio
    best_r_threshold = stats_train._strategy.r_threshold

    print(
        f"بهترین پارامترها -> n_swing={best_n_swing}, "
        f"rr_ratio={best_rr_ratio}, r_threshold={best_r_threshold}"
    )

    # --------------  test با پارامترهای ثابت ----------
    class TunedStrategy(AdvancedShadowStrategy):
        n_swing = best_n_swing
        rr_ratio = best_rr_ratio
        r_threshold = best_r_threshold

    bt_test = Backtest(
        test_df,
        TunedStrategy,
        cash=1_000_000,
        commission=0.0001,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats_test = bt_test.run()

    print(f"Test Return [%]: {stats_test['Return [%]']:.2f}")

    # ---------- ذخیره‌ی خلاصه‌ی این پنجره ----------
    summary_rows.append(
        {
            "window": window_id,
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "best_n_swing": best_n_swing,
            "best_rr_ratio": best_rr_ratio,
            "best_r_threshold": best_r_threshold,
            "train_return_pct": stats_train["Return [%]"],
            "test_return_pct": stats_test["Return [%]"],
            "test_trades": stats_test["# Trades"],
            "test_win_rate_pct": stats_test["Win Rate [%]"],
            "test_max_drawdown_pct": stats_test["Max. Drawdown [%]"],
        }
    )

    # ---------- تجمیع معاملات و برخوردهای test هر پنجره ----------
    test_trades_df = stats_test["_trades"].copy()
    test_trades_df["window"] = window_id
    all_test_trades.append(test_trades_df)

    test_touches_df = pd.DataFrame(stats_test._strategy.touch_log)
    if not test_touches_df.empty:
        test_touches_df["window"] = window_id
        all_test_touches.append(test_touches_df)


#%%
# ==========================================
# ذخیره‌ی خروجی نهایی
# ==========================================

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel("walk_forward_summary.xlsx", index=False)
print(f"\n{len(summary_df)} پنجره در walk_forward_summary.xlsx ذخیره شد")

if all_test_trades:
    combined_trades = pd.concat(all_test_trades, ignore_index=True)
    combined_trades.to_excel("walk_forward_test_trades.xlsx", index=False)
    print(f"{len(combined_trades)} معامله (فقط از بازه‌های test) در walk_forward_test_trades.xlsx ذخیره شد")

if all_test_touches:
    combined_touches = pd.concat(all_test_touches, ignore_index=True)
    combined_touches.to_excel("walk_forward_test_zone_touches.xlsx", index=False)
    print(f"{len(combined_touches)} برخورد (فقط از بازه‌های test) در walk_forward_test_zone_touches.xlsx ذخیره شد")

print("\n===== خلاصه‌ی نهایی =====")
print(summary_df[["window", "train_return_pct", "test_return_pct", "test_trades", "test_win_rate_pct"]])

print("\nDone!")
