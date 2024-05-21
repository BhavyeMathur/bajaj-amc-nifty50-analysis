import datetime
import pprint

import yfinance as yf
import pandas as pd
import numpy as np

today = datetime.datetime.now()


class Stock(yf.Ticker):
    def __init__(self, name: str):
        super().__init__(name)
        self._name = name

    def print_info(self) -> None:
        pprint.pprint(self.info)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Stock):
            return self.name == o.name
        return False

    def __hash__(self) -> int:
        return hash(self._name)

    def history(self, period="1mo", interval="1d", start=None, end=None) -> pd.DataFrame:
        d = super().history(period=period, interval=interval, start=start, end=end)
        d["Change"] = d["Close"].diff()
        d["Prev Close"] = d["Close"] - d["Change"]
        d["% Change"] = d["Change"] / d["Prev Close"]
        d["Gross % Change"] = d["% Change"] + 1
        return d

    def historical_returns(self, period="max", interval="1mo", start=None, end=None) -> np.ndarray:
        d = self.history(period=period, interval=interval, start=start, end=end)
        return np.array([d["Gross % Change"].mean() - 1, d["% Change"].std()])

    @property
    def name(self) -> str:
        return self._name
