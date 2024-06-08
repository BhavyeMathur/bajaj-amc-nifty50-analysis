from __future__ import annotations

import pandas as pd
import numpy as np


class Ticker:
    def __init__(self, ticker: str):
        self._symbol = ticker
        self._name = ticker

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Ticker):
            return self._symbol == o.symbol
        return False

    def __hash__(self) -> int:
        return hash(self._symbol)

    def print_info(self) -> None:
        print(self)

    def history(self, period="1mo", interval="1d", start=None, end=None) -> pd.DataFrame:
        """
        Returns historical data on a ticker with columns "Open", "High", "Low", "Close", "Volume", "Prev Close",
        "Change", "% Change", and "Gross % Change"
        """
        raise NotImplementedError()

    def historical_returns(self, period="max", interval="1mo", start=None, end=None) -> np.ndarray:
        """
        Returns the mean historical % return and standard deviation of a ticker
        """
        d = self.history(period=period, interval=interval, start=start, end=end)
        return np.array([d["% Change"].mean(), d["% Change"].std()])

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def currency(self) -> str:
        raise NotImplementedError()


__all__ = ["Ticker"]
