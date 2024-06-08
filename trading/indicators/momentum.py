from ta import momentum
import pandas as pd


from .abstract import MomentumIndicator


class AwesomeOscillatorIndicator(MomentumIndicator):
    def __init__(self, window1: int = 5, window2: int = 34):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.awesome_oscillator(data["High"], data["Low"], window1, window2)
        self.__call__ = call


class KAMAIndicator(MomentumIndicator):
    def __init__(self, window: int = 10, pow1: int = 2, pow2: int = 30):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.kama(data["Close"], window, pow1, pow2)
        self.__call__ = call


class PercentagePriceIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.ppo(data["Close"], window_slow, window_fast, window_sign)
        self.__call__ = call


class PercentageVolumeIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.pvo(data["Close"], window_slow, window_fast, window_sign)
        self.__call__ = call


class ROCIndicator(MomentumIndicator):
    def __init__(self, window: int = 12):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.roc(data["Close"], window)
        self.__call__ = call


class RSIIndicator(MomentumIndicator):
    def __init__(self, window: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.rsi(data["Close"], window)
        self.__call__ = call


class StochRSIIndicator(MomentumIndicator):
    def __init__(self, window: int = 14, smooth1: int = 3, smooth2: int = 3):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.stochrsi(data["Close"], window, smooth1, smooth2)
        self.__call__ = call


class StochasticOscillatorIndicator(MomentumIndicator):
    def __init__(self, window: int = 14, smooth_window: int = 3):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.stoch(data["High"], data["Low"], data["Close"], window, smooth_window)
        self.__call__ = call


class TSIIndicator(MomentumIndicator):
    def __init__(self, window_slow: int = 25, window_fast: int = 13):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.stochrsi(data["Close"], window_slow, window_fast)
        self.__call__ = call


class UltimateOscillatorIndicator(MomentumIndicator):
    def __init__(self, window1: int = 7, window2: int = 14, window3: int = 28,
                 weight1: float = 4.0, weight2: float = 2.0, weight3: float = 1.0):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.ultimate_oscillator(data["High"], data["Low"], data["Close"],
                                                window1, window2, window3, weight1, weight2, weight3)
        self.__call__ = call


class WilliamsRIndicator(MomentumIndicator):
    def __init__(self, lbp: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return momentum.williams_r(data["High"], data["Low"], data["Close"], lbp)
        self.__call__ = call


__all__ = ["AwesomeOscillatorIndicator", "KAMAIndicator", "PercentagePriceIndicator", "PercentageVolumeIndicator",
           "ROCIndicator", "RSIIndicator", "StochRSIIndicator", "StochasticOscillatorIndicator", "TSIIndicator",
           "UltimateOscillatorIndicator", "WilliamsRIndicator"]
