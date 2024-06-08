from ta import momentum
import pandas as pd

from .abstract import MomentumIndicator


class AwesomeOscillatorIndicator(MomentumIndicator):
    def __init__(self, window1: int = 5, window2: int = 34):
        self.window1 = window1
        self.window2 = window2

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.awesome_oscillator(data["High"], data["Low"], self.window1, self.window2)


class KAMAIndicator(MomentumIndicator):
    def __init__(self, window: int = 10, pow1: int = 2, pow2: int = 30):
        self.window = window
        self.pow1 = pow1
        self.pow2 = pow2

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.kama(data["Close"], self.window, self.pow1, self.pow2)


class PercentagePriceIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.ppo(data["Close"], self.window_slow, self.window_fast, self.window_sign)


class PercentageVolumeIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.pvo(data["Close"], self.window_slow, self.window_fast, self.window_sign)


class ROCIndicator(MomentumIndicator):
    def __init__(self, window: int = 12):
        self.window = window

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.roc(data["Close"], self.window)


class RSIIndicator(MomentumIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.rsi(data["Close"], self.window)


class StochRSIIndicator(MomentumIndicator):
    def __init__(self, window: int = 14, smooth1: int = 3, smooth2: int = 3):
        def __call__(self, data: pd.DataFrame) -> pd.Series:
            return momentum.stochrsi(data["Close"], window, smooth1, smooth2)


class StochasticOscillatorIndicator(MomentumIndicator):
    def __init__(self, window: int = 14, smooth_window: int = 3):
        self.window = window
        self.smooth_window = smooth_window

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.stoch(data["High"], data["Low"], data["Close"], self.window, self.smooth_window)


class TSIIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 25, window_fast: int = 13):
        self.window_slow = window_slow
        self.window_fast = window_fast

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.stochrsi(data["Close"], self.window_slow, self.window_fast)


class UltimateOscillatorIndicator(MomentumIndicator):
    def __init__(self, window1: int = 7, window2: int = 14, window3: int = 28,
                 weight1: float = 4.0, weight2: float = 2.0, weight3: float = 1.0):
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3

        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.ultimate_oscillator(data["High"], data["Low"], data["Close"],
                                            self.window1, self.window2, self.window3,
                                            self.weight1, self.weight2, self.weight3)


class WilliamsRIndicator(MomentumIndicator):
    def __init__(self, lbp: int = 14):
        self.lbp = lbp

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return momentum.williams_r(data["High"], data["Low"], data["Close"], self.lbp)


__all__ = ["AwesomeOscillatorIndicator", "KAMAIndicator", "PercentagePriceIndicator", "PercentageVolumeIndicator",
           "ROCIndicator", "RSIIndicator", "StochRSIIndicator", "StochasticOscillatorIndicator", "TSIIndicator",
           "UltimateOscillatorIndicator", "WilliamsRIndicator"]
