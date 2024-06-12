import os
import pickle
import pprint

import pandas as pd
import yfinance as yf

from .ticker import Ticker


class YTicker(Ticker):
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self._yticker = yf.Ticker(ticker)

    def print_info(self):
        pprint.pprint(self._yticker.info)

    def history(self, period="1mo", interval="1d", start=None, end=None) -> pd.DataFrame:
        path = f"data/{self._symbol}-{period}-{interval}-{start}-{end}.bin"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        d = self._yticker.history(period=period, interval=interval, start=start, end=end)
        d["Change"] = d["Close"].diff()
        d["Prev Close"] = d["Close"] - d["Change"]
        d["% Change"] = d["Change"] / d["Prev Close"]
        d["Gross % Change"] = d["% Change"] + 1

        with open(path, "wb") as f:
            pickle.dump(d, f)
        return d

    @property
    def currency(self) -> str:
        return self._yticker.info["currency"]

    @property
    def name(self) -> str:
        return self._yticker.info["shortName"]


__all__ = ["YTicker"]
