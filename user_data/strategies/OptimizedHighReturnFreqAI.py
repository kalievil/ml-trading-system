from functools import reduce
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


class OptimizedHighReturnFreqAI(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count: int = 200

    minimal_roi = {"0": 0.015}
    stoploss = -0.018
    use_exit_signal = True

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {"do_predict": {"color": "brown"}},
            "target": {"&-cls": {"color": "blue"}},
        },
    }

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        macd = ta.MACD(dataframe)
        dataframe["%-macd-period"] = macd["macd"]
        dataframe["%-macdsignal-period"] = macd["macdsignal"]
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        dataframe["bb_lowerband-period"] = bb["lower"]
        dataframe["bb_middleband-period"] = bb["mid"]
        dataframe["bb_upperband-period"] = bb["upper"]
        dataframe["%-bb_width-period"] = (dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]) / dataframe["bb_middleband-period"]
        dataframe["%-volume_ratio-period"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        dataframe["%-returns-period"] = dataframe["close"].pct_change()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        # Classification target with your original 1.5% take-profit threshold
        # 1 = long, 2 = short, 0 = hold
        future_1 = dataframe["close"].shift(-1) / dataframe["close"] - 1
        future_2 = dataframe["close"].shift(-2) / dataframe["close"] - 1
        avg_ret = (future_1 + future_2) / 2
        target = np.where(avg_ret > 0.015, 1, np.where(avg_ret < -0.015, 2, 0))
        dataframe["&-cls"] = target
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Use model output: 'do_predict' plus class prediction thresholds
        # FreqAI will add columns like '&-cls_pred' and maybe proba depending on model
        long_conditions = [df["do_predict"] == 1, df.get("&-cls_pred", 0) == 1]
        if long_conditions:
            df.loc[reduce(lambda x, y: x & y, long_conditions), ["enter_long", "enter_tag"]] = (1, "long")

        short_conditions = [df["do_predict"] == 1, df.get("&-cls_pred", 0) == 2]
        if short_conditions:
            df.loc[reduce(lambda x, y: x & y, short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Exit when class flips back to 0 (hold) or opposite signal appears
        exit_long = [df["do_predict"] == 1, df.get("&-cls_pred", 0) != 1]
        if exit_long:
            df.loc[reduce(lambda x, y: x & y, exit_long), "exit_long"] = 1

        exit_short = [df["do_predict"] == 1, df.get("&-cls_pred", 0) != 2]
        if exit_short:
            df.loc[reduce(lambda x, y: x & y, exit_short), "exit_short"] = 1

        return df
