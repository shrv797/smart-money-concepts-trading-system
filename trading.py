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

def calc_ema(series, period):
    return (
        pd.Series(series)
        .ewm(span=period, adjust=False)
        .mean()
        .values
    )
# در این تابع میانگین متحرک نمایی محاسبه میشود
#قیمت بالای ای ام ای روند صعودی و پایین روند نزولی ست

def calc_atr(high, low, close, period):
    """
    Wilder ATR
    """

    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr.values

# با کمک این تابع و ای تی آر نوسانات بازار را اندازه گیری کردیم
#برای استاپ لاس و فسلتر کردن کندل های خیلی کوچک
# ==========================================
# Strategy
# ==========================================

class AdvancedShadowStrategy(Strategy):

    # ---------- Parameters ----------
# پارامتر ها ذر استراتژی که تعدادی از ان ها بهینه سازی شدند
    n_swing =  3 #5

    tolerance = 0.002

    ema_period = 200

    atr_period = 14

    rr_ratio = 2.0

    r_threshold = 2.0

    max_zones = 30

    min_zone_touches = 2

    body_atr_filter = 0.10

    # --------------------------------

    def init(self):
# محاسبات لازم را انجام می دهیم در این تابع
        self.ema = self.I(
            calc_ema,
            self.data.Close,
            self.ema_period
        )

        self.atr = self.I(
            calc_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
        )

        # Swing ها

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
#با این تابع نواحی مقاومت و حمایت را کنترل میکنیم
    def update_zones(
        self,
        new_price,
        zones,
        index
    ):

        merged = False

        for zone in zones:

            if abs(
                new_price - zone["level"]
            ) / zone["level"] <= self.tolerance:

                zone["prices"].append(new_price)

                zone["level"] = np.mean(
                    zone["prices"]
                )

                zone["low"] = min(
                    zone["prices"]
                )

                zone["high"] = max(
                    zone["prices"]
                )

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

        # فقط آخرین نواحی نگهداری شوند

        if len(zones) > self.max_zones:

            zones.pop(0)

    # ==========================================

    def zone_available(self, level):
#بررسی ناحیه برای انجام معامله
        if self.last_trade_zone is None:
            return True

        distance = abs(
            level - self.last_trade_zone
        ) / level

        return distance > self.tolerance

    # ==========================================

    def next(self):

        idx = len(self.data) - 1

        n = self.n_swing

        if (
            idx < 2 * n + 5
            or np.isnan(self.ema[-1])
            or np.isnan(self.atr[-1])
        ):
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

        if (
            is_high
            and swing_index not in self.swing_high_index
        ):

            self.swing_high_index.add(
                swing_index
            )

            self.update_zones(
                candidate_h,
                self.res_zones,
                swing_index
            )

            self.swing_highs.append(
                {
                    "index": swing_index,
                    "price": candidate_h,
                }
            )
        if (
            is_low
            and swing_index not in self.swing_low_index
        ):

            self.swing_low_index.add(
                swing_index
            )

            self.update_zones(
                candidate_l,
                self.sup_zones,
                swing_index
            )

            self.swing_lows.append(
                {
                    "index": swing_index,
                    "price": candidate_l,
                }
        )
                    # ======================================
        # Candle Variables
        # ======================================

        close = self.data.Close[-1]
        open_ = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        ema = self.ema[-1]
        atr = self.atr[-1]

        body = abs(close - open_)

        # جلوگیری از سیگنال‌های ناشی از دوجی‌های خیلی کوچک
        if body < atr * self.body_atr_filter:
            return

        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low

        r_upper = upper_shadow / body
        r_lower = lower_shadow / body

        bullish = close > open_
        bearish = close < open_

        # اگر معامله باز داریم، معامله جدید باز نکن
        if self.position:
            return


        #Trend
        """
        if len(self.swing_highs) >= 3 and len(self.swing_lows) >= 3:
            last3 = self.swing_highs[-3:]
            last3_l = self.swing_lows[-3:]
            if last3[-3]["price"] < last3[-2]["price"] < last3[-1]["price"]:
                if last3_l[-3]["price"] < last3_l[-2]["price"] < last3_l[-1]["price"]:
                    self.market_trend = 'bull'
            if last3[-3]["price"] > last3[-2]["price"] > last3[-1]["price"]:
                if last3_l[-3]["price"] > last3_l[-2]["price"] > last3_l[-1]["price"]:
                    self.market_trend = 'bear'
        """

        # ======================================
        # BOS and CHOCH
        # BOS

        bullish_bos = False
        bearish_bos = False

        if len(self.swing_highs) >= 2:
            reference_high = self.swing_highs[-2]["price"]
            bullish_bos = close > reference_high

        if len(self.swing_lows) >= 2:
            reference_low = self.swing_lows[-2]["price"]
            bullish_bos = close > reference_low

        #CHOCH
        """
        choch = None
        if self.market_trend == "bull" and self.last_swing_low is not None :
            if close < self.last_swing_low:
                choch = "Bearish CHoCH"
                self.market_trend = "bear"
        elif ((self.market_trend == "bear") and (self.last_swing_high is not None)):
            if close > self.last_swing_high:
                choch = "Bullish CHoCH"
                self.market_trend = "bull"
                
        """

        # ======================================
        # SHORT
        # ======================================

        if bearish_bos and close < ema and r_upper >= self.r_threshold:

            for zone in self.res_zones:

                # فقط مقاومت‌های معتبر
                if zone["touches"] < self.min_zone_touches:
                    continue

                z_low = zone["low"]
                z_high = zone["high"]
                z_mid = zone["level"]

                # قبلاً روی این ناحیه معامله شده؟
                if not self.zone_available(z_mid):
                    continue

                # برخورد واقعی به مقاومت
                if (
                    high >= z_low
                    and high <= z_high + atr
                    and close < z_mid
                ):

                    sl = max(
                        z_high + atr,
                        high + atr * 0.20
                    )

                    risk = sl - close

                    if risk <= 0:
                        continue

                    tp = close - risk * self.rr_ratio

                    self.sell(
                        sl=sl,
                        tp=tp
                    )

                    self.last_trade_zone = z_mid

                    return

        # ======================================
        # LONG
        # ======================================

        if bullish_bos and close > ema and r_lower >= self.r_threshold:

            for zone in self.sup_zones:

                # فقط حمایت‌های معتبر
                if zone["touches"] < self.min_zone_touches:
                    continue

                z_low = zone["low"]
                z_high = zone["high"]
                z_mid = zone["level"]

                if not self.zone_available(z_mid):
                    continue

                if (
                    low <= z_high
                    and low >= z_low - atr
                    and close > z_mid
                ):

                    sl = min(
                        z_low - atr,
                        low - atr * 0.20
                    )

                    risk = close - sl

                    if risk <= 0:
                        continue

                    tp = close + risk * self.rr_ratio

                    self.buy(
                        sl=sl,
                        tp=tp
                    )

                    self.last_trade_zone = z_mid

                    return
                    # ==========================================
# اجرای بک تست
# ==========================================

if __name__ == "__main__":

    try:

        df = pd.read_csv("C:\\Users\\pc\\Desktop\\data\\btc-opt.csv")

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"]
        )

        df.set_index(
            "Timestamp",
            inplace=True
        )

        # حذف ستون‌های اضافی

        for col in ["Unnamed: 0", "index"]:

            if col in df.columns:
                df.drop(
                    col,
                    axis=1,
                    inplace=True
                )

        # اطمینان از نوع داده‌ها

        cols = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for c in cols:
            df[c] = df[c].astype(float)

        # اگر حجم وجود دارد

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)

    except FileNotFoundError:

        print("CSV File Not Found")

        exit()

    except Exception as e:

        print(e)

        exit()

    # ======================================

    bt = Backtest(

        df,

        AdvancedShadowStrategy,

        cash=1_000_000,

        commission=0.0001,

        trade_on_close=True,

        exclusive_orders=True

    )

    stats = bt.run()

    print("\n========== RESULT ==========\n")

    print(stats)

    print("\n========== TRADES ==========\n")

    info = stats["_trades"]

    print(info)

    # ------------------------

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
            sh.write(i+1, indx2 - indx, str(info[col][i]))
            indx -= 1
        indx = len(columns)

    wb.save('trade_info.xlsx')

    # نمودار

# stats = bt.optimize(

#     n_swing=[3,5,7],

#     ema_period=[100,150,200],

#     rr_ratio=[1.5,2],

#     maximize='Return [%]'

# )
# print(stats)
# print("\nBest Parameters:")
# print(stats._strategy)
# print("\nTrades:")
# print(stats["# Trades"])
