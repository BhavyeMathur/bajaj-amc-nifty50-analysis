from ta import volume
import pandas as pd

from .abstract import VolumeIndicator


class AccDistIndexIndicator(VolumeIndicator):
    def __call__(self, data) -> pd.Series:
        return volume.acc_dist_index(data["High"], data["Low"], data["Close"], data["Volume"])


class ChaikinMoneyFlowIndicator(VolumeIndicator):
    def __init__(self, window: int = 20):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volume.chaikin_money_flow(data["High"], data["Low"], data["Close"], data["Volume"], self.window)


class EaseOfMovementIndicator(VolumeIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volume.ease_of_movement(data["High"], data["Low"], data["Volume"], self.window)


class ForceIndexIndicator(VolumeIndicator):
    def __init__(self, window: int = 13):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volume.force_index(data["Close"], data["Volume"], self.window)


class MoneyFlowIndexIndicator(VolumeIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volume.money_flow_index(data["High"], data["Low"], data["Close"], data["Volume"], self.window)


class NegativeVolumeIndexIndicator(VolumeIndicator):
    def __call__(self, data) -> pd.Series:
        return volume.negative_volume_index(data["Close"], data["Volume"])


class OnBalanceVolumeIndicator(VolumeIndicator):
    def __call__(self, data) -> pd.Series:
        return volume.on_balance_volume(data["Close"], data["Volume"])


class VolumePriceTrendIndicator(VolumeIndicator):
    def __call__(self, data) -> pd.Series:
        return volume.volume_price_trend(data["Close"], data["Volume"])


class VolumeWeightedAveragePriceIndicator(VolumeIndicator):
    def __init__(self, window: int = 14):
        self.window = window

    def __call__(self, data) -> pd.Series:
        return volume.volume_weighted_average_price(data["High"], data["Low"], data["Close"], data["Volume"],
                                                    self.window)


__all__ = ["AccDistIndexIndicator", "ChaikinMoneyFlowIndicator", "EaseOfMovementIndicator", "ForceIndexIndicator",
           "MoneyFlowIndexIndicator", "NegativeVolumeIndexIndicator", "OnBalanceVolumeIndicator",
           "VolumePriceTrendIndicator", "VolumeWeightedAveragePriceIndicator"]
