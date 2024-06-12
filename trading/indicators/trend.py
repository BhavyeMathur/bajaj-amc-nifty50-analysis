from ta import trend
import pandas as pd

from .abstract import TrendIndicator


class ADXIndicator(TrendIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.adx(data["High"], data["Low"], data["Close"], self.window)


class AroonDownIndicator(TrendIndicator):
    def __init__(self, window: int = 25):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.aroon_down(data["High"], data["Low"], self.window)


class AroonUpIndicator(TrendIndicator):
    def __init__(self, window: int = 25):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.aroon_up(data["High"], data["Low"], self.window)


class CCIIndicator(TrendIndicator):
    def __init__(self, window: int = 20, constant: float = 0.015):
        self.window = window
        self.constant = constant

    def __call__(self, data) -> pd.Series:
        return trend.cci(data["High"], data["Low"], data["Close"], self.window, self.constant)


class DPOIndicator(TrendIndicator):
    def __init__(self, window: int = 20):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.dpo(data["Close"], self.window)


class EMAIndicator(TrendIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.ema_indicator(data["Close"], self.window)


class KSTIndicator(TrendIndicator):
    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                 window1: int = 10, window2: int = 10, window3: int = 10, window4: int = 15):
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4

        self.window1 = window1
        self.window2 = window2
        self.window3 = window3
        self.window4 = window4

    def __call__(self, data) -> pd.Series:
        return trend.kst(data["Close"], self.roc1, self.roc2, self.roc3, self.roc4,
                         self.window1, self.window2, self.window3, self.window4)


class MACDIndicator(TrendIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12):
        self.window_slow = window_slow
        self.window_fast = window_fast

    def __call__(self, data) -> pd.Series:
        return trend.macd(data["Close"], self.window_slow, self.window_fast)


class MassIndexIndicator(TrendIndicator):
    def __init__(self, window_fast: int = 9, window_slow: int = 25):
        self.window_fast = window_fast
        self.window_slow = window_slow

    def __call__(self, data) -> pd.Series:
        return trend.mass_index(data["High"], data["Low"], self.window_fast, self.window_slow)


class SMAIndicator(TrendIndicator):
    def __init__(self, window: int = 12):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.sma_indicator(data["Close"], self.window)


class STCIndicator(TrendIndicator):
    def __init__(self, window_slow: int = 50, window_fast: int = 23, cycle: int = 10,
                 smooth1: int = 3, smooth2: int = 3):
        self.window_slow = window_slow
        self.window_fast = window_fast

        self.cycle = cycle
        self.smooth1 = smooth1
        self.smooth2 = smooth2

    def __call__(self, data) -> pd.Series:
        return trend.stc(data["Close"], self.window_slow, self.window_fast, self.cycle, self.smooth1, self.smooth2)


class TRIXIndicator(TrendIndicator):
    def __init__(self, window: int = 15):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.trix(data["Close"], self.window)


class WMAIndicator(TrendIndicator):
    def __init__(self, window: int = 9):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return trend.wma_indicator(data["Close"], self.window)


__all__ = ["ADXIndicator", "AroonUpIndicator", "AroonDownIndicator", "CCIIndicator", "DPOIndicator", "EMAIndicator",
           "KSTIndicator", "MACDIndicator", "MassIndexIndicator", "SMAIndicator", "STCIndicator", "TRIXIndicator",
           "WMAIndicator"]
