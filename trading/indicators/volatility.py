from ta import volatility
import pandas as pd

from .abstract import VolatilityIndicator


class AverageTrueRangeIndicator(VolatilityIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volatility.average_true_range(data["High"], data["Low"], data["Close"], self.window)


class BollingerBands(VolatilityIndicator):
    def __init__(self, window: int = 20, window_dev: int = 2):
        self.window = window
        self.window_dev = window_dev

    def __call__(self, data) -> tuple[pd.Series, pd.Series]:
        return (volatility.bollinger_lband(data["Close"], self.window, self.window_dev),
                volatility.bollinger_hband(data["Close"], self.window, self.window_dev))


class DonchianChannels(VolatilityIndicator):
    def __init__(self, window: int = 20, offset: int = 0):
        self.window = window
        self.offset = offset

    def __call__(self, data) -> tuple[pd.Series, pd.Series]:
        return (volatility.donchian_channel_lband(data["High"], data["Low"], data["Close"], self.window, self.offset),
                volatility.donchian_channel_hband(data["High"], data["Low"], data["Close"], self.window, self.offset))


class KeltnerChannels(VolatilityIndicator):
    def __init__(self, window: int = 20, window_atr: int = 10):
        self.window = window
        self.window_atr = window_atr

    def __call__(self, data) -> tuple[pd.Series, pd.Series]:
        return (
            volatility.keltner_channel_hband(data["High"], data["Low"], data["Close"], self.window, self.window_atr),
            volatility.keltner_channel_hband(data["High"], data["Low"], data["Close"], self.window, self.window_atr))


class UlcerIndexIndicator(VolatilityIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volatility.ulcer_index(data["Close"], self.window)


__all__ = ["AverageTrueRangeIndicator", "BollingerBands", "DonchianChannels", "KeltnerChannels", "UlcerIndexIndicator"]
