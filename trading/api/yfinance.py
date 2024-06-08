import pprint

import yfinance as yf

from .ticker import Ticker


class YTicker(Ticker):
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self._yticker = yf.Ticker(ticker)
        self._name = self._yticker.info["shortName"]

    def print_info(self):
        pprint.pprint(self._yticker.info)

    def history(self, period="1mo", interval="1d", start=None, end=None):
        d = self._yticker.history(period=period, interval=interval, start=start, end=end)
        d["Change"] = d["Close"].diff()
        d["Prev Close"] = d["Close"] - d["Change"]
        d["% Change"] = d["Change"] / d["Prev Close"]
        d["Gross % Change"] = d["% Change"] + 1
        return d

    @property
    def currency(self) -> str:
        return self._yticker.info["currency"]


__all__ = ["YTicker"]
