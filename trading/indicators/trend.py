from ta import trend
import pandas as pd


from .abstract import TrendIndicator


class ADXIndicator(TrendIndicator):
    def __init__(self, window: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.adx(data["High"], data["Low"], data["Close"], window)
        self.__call__ = call


class AroonDownIndicator(TrendIndicator):
    def __init__(self, window: int = 25):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.aroon_down(data["High"], data["Low"], window)
        self.__call__ = call


class AroonUpIndicator(TrendIndicator):
    def __init__(self, window: int = 25):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.aroon_up(data["High"], data["Low"], window)
        self.__call__ = call


class CCIIndicator(TrendIndicator):
    def __init__(self, window: int = 20, constant: float = 0.015):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.cci(data["High"], data["Low"], data["Close"], window, constant)
        self.__call__ = call


class DPOIndicator(TrendIndicator):
    def __init__(self, window: int = 20):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.dpo(data["Close"], window)
        self.__call__ = call


class EMAIndicator(TrendIndicator):
    def __init__(self, window: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.ema_indicator(data["Close"], window)
        self.__call__ = call


class KSTIndicator(TrendIndicator):
    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                 window1: int = 10, window2: int = 10, window3: int = 10, window4: int = 15):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.kst(data["Close"], roc1, roc2, roc3, roc4, window1, window2, window3, window4)
        self.__call__ = call


class MACDIndicator(TrendIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.macd(data["Close"], window_slow, window_fast)
        self.__call__ = call


class MassIndexIndicator(TrendIndicator):
    def __init__(self, window_fast: int = 9, window_slow: int = 25):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.mass_index(data["High"], data["Low"], window_fast, window_slow)
        self.__call__ = call


class SMAIndicator(TrendIndicator):
    def __init__(self, window: int = 12):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.sma_indicator(data["Close"], window)
        self.__call__ = call


class STCIndicator(TrendIndicator):
    def __init__(self, window_slow: int = 50, window_fast: int = 23, cycle: int = 10,
                 smooth1: int = 3, smooth2: int = 3):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.stc(data["Close"], window_slow, window_fast, cycle, smooth1, smooth2)
        self.__call__ = call


class TRIXIndicator(TrendIndicator):
    def __init__(self, window: int = 15):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.trix(data["Close"], window)
        self.__call__ = call


class WMAIndicator(TrendIndicator):
    def __init__(self, window: int = 9):
        def call(data: pd.DataFrame) -> pd.Series:
            return trend.wma_indicator(data["Close"], window)
        self.__call__ = call


__all__ = ["ADXIndicator", "AroonUpIndicator", "AroonDownIndicator", "CCIIndicator", "DPOIndicator", "EMAIndicator",
           "KSTIndicator", "MACDIndicator", "MassIndexIndicator", "SMAIndicator", "STCIndicator", "TRIXIndicator",
           "WMAIndicator"]
