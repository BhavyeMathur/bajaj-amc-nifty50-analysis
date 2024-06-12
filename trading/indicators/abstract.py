import pandas as pd


class Indicator:
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()


class MomentumIndicator(Indicator):
    pass


class VolumeIndicator(Indicator):
    pass


class VolatilityIndicator(Indicator):
    pass


class TrendIndicator(Indicator):
    pass
