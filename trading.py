import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import warnings
import xlwt as xt

warnings.filterwarnings("ignore")
# حذف و مخفی کردن پیام های غیرضروری
# ==========================================
# اندیکاتورها
# ==========================================

# لیست سراسری برای ثبت تمام برخوردهای قیمت با نواحی حمایت و مقاومت
ZONE_TOUCH_LOG = []


class AdvancedShadowStrategy(Strategy):

    # ---------- Parameters ----------
    n_swing = 3  # 5

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
        """
        ثبت هر برخورد کندل جاری با یک ناحیه حمایت/مقاومت
        """

        ZONE_TOUCH_LOG.append(
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
        # (این بخش مستقل از باز بودن معامله یا سیگنال اجرا میشود
        #  تا تمام برخوردها ثبت شوند، نه فقط آنهایی که منجر به معامله میشوند)
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
        # BOS and CHOCH
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

                    # به‌روزرسانی آخرین رکورد ثبت‌شده برای این ناحیه: معامله انجام شد
                    if ZONE_TOUCH_LOG:
                        for rec in reversed(ZONE_TOUCH_LOG):
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

                    if ZONE_TOUCH_LOG:
                        for rec in reversed(ZONE_TOUCH_LOG):
                            if (
                                rec["time"] == self.data.index[-1]
                                and rec["zone_type"] == "support"
                                and rec["zone_level"] == z_mid
                            ):
                                rec["trade_taken"] = True
                                break

                    return


# ==========================================
# اجرای بک تست
# ==========================================

if __name__ == "__main__":

    try:

        df = pd.read_csv("btc-opt (1).csv")

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)

        for col in ["Unnamed: 0", "index"]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        cols = ["Open", "High", "Low", "Close"]

        for c in cols:
            df[c] = df[c].astype(float)

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

    except FileNotFoundError:

        print("CSV File Not Found")
        exit()

    except Exception as e:

        print(e)
        exit()

    # پاک کردن لاگ نواحی قبل از هر اجرای جدید بک تست
    ZONE_TOUCH_LOG.clear()

    bt = Backtest(
        df,
        AdvancedShadowStrategy,
        cash=1_000_000,
        commission=0.0001,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats = bt.run()

    print("\n========== RESULT ==========\n")
    print(stats)

    print("\n========== TRADES ==========\n")

    info = stats["_trades"]
    print(info)

    # ------------------------
    # خروجی معاملات (اکسل قدیمی با xlwt)

    columns = list(stats["_trades"].columns)
    wb = xt.Workbook()
    sh = wb.add_sheet('btc_opt')
    indx = len(columns)
    indx2 = len(columns)
    for coll in columns:
        sh.write(0, indx2 - indx, str(coll))
        indx -= 1
    indx = len(columns)

    for i in range(len(info)):
        for col in columns:
            sh.write(i + 1, indx2 - indx, str(info[col][i]))
            indx -= 1
        indx = len(columns)

    wb.save('trade_info.xlsx')

    # ------------------------
    # خروجی معاملات (فرمت جدید)

    trades = stats["_trades"]
    trades.to_excel("opt result.xlsx", index=False)

    # ------------------------
    # خروجی ثبت برخوردها با نواحی حمایت و مقاومت

    zones_df = pd.DataFrame(ZONE_TOUCH_LOG)
    zones_df.to_excel("zone_touches.xlsx", index=False)

    print(f"\nتعداد برخوردهای ثبت‌شده با نواحی: {len(ZONE_TOUCH_LOG)}")
    print("Saved!")
