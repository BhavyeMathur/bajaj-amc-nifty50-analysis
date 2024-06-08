from ta import volatility
import pandas as pd


from .abstract import VolatilityIndicator


class AverageTrueRangeIndicator(VolatilityIndicator):
    def __init__(self, window: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.average_true_range(data["High"], data["Low"], data["Close"], window)
        self.__call__ = call


class BollingerHighBand(VolatilityIndicator):
    def __init__(self, window: int = 20, window_dev: int = 2):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.bollinger_hband(data["Close"], window, window_dev)
        self.__call__ = call


class BollingerLowBand(VolatilityIndicator):
    def __init__(self, window: int = 20, window_dev: int = 2):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.bollinger_lband(data["Close"], window, window_dev)
        self.__call__ = call


class DonchianHighChannel(VolatilityIndicator):
    def __init__(self, window: int = 20, offset: int = 0):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.donchian_channel_hband(data["High"], data["Low"], data["Close"], window, offset)
        self.__call__ = call


class DonchianLowChannel(VolatilityIndicator):
    def __init__(self, window: int = 20, offset: int = 0):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.donchian_channel_lband(data["High"], data["Low"], data["Close"], window, offset)
        self.__call__ = call


class KeltnerHighChannel(VolatilityIndicator):
    def __init__(self, window: int = 20, window_atr: int = 10):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.keltner_channel_hband(data["High"], data["Low"], data["Close"], window, window_atr)
        self.__call__ = call


class KeltnerLowChannel(VolatilityIndicator):
    def __init__(self, window: int = 20, window_atr: int = 10):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.keltner_channel_hband(data["High"], data["Low"], data["Close"], window, window_atr)
        self.__call__ = call


class UlcerIndexIndicator(VolatilityIndicator):
    def __init__(self, window: int = 14):
        def call(data: pd.DataFrame) -> pd.Series:
            return volatility.ulcer_index(data["Close"], window)
        self.__call__ = call


__all__ = ["AverageTrueRangeIndicator", "BollingerLowBand", "BollingerHighBand",
           "DonchianHighChannel", "DonchianLowChannel", "KeltnerHighChannel", "KeltnerLowChannel",
           "UlcerIndexIndicator"]
