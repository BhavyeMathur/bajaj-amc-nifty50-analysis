import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

import pandas as pd

from trading import Ticker


class TickerPlot:
    def __init__(self, ticker: Ticker):
        self._ticker = ticker
        self._fig: None | plt.Figure = None
        self._ax: None | plt.Axes = None

    def _plot(self, data: pd.DataFrame) -> None:
        raise NotImplementedError()

    def show(self, period="1mo", interval="1d", start=None, end=None) -> None:
        self._fig = plt.figure(figsize=(9, 4))
        self._ax = self._fig.gca()
        self._fig.suptitle(self._ticker.name, fontsize=10, y=0.95)

        self._ax.xaxis.set_major_formatter(mdates.DateFormatter("%b, %Y"))
        self._ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f} " + self._ticker.currency))
        self._ax.xaxis.set_tick_params(labelsize=6)
        self._ax.yaxis.set_tick_params(labelsize=6)

        data = self._ticker.history(period=period, interval=interval, start=start, end=end)
        self._plot(data)

        plt.show()


class CandlestickPlot(TickerPlot):
    def _plot(self, data: pd.DataFrame) -> None:
        up = data[data["Close"] > data["Prev Close"]]
        down = data[data["Close"] <= data["Prev Close"]]

        w = 1056.6 / (len(data) ** 1.46508)
        self._ax.vlines(up.index, up["Open"], up["Close"], color="green", linestyle="-", lw=w)
        self._ax.vlines(down.index, down["Open"], down["Close"], color="red", linestyle="-", lw=w)

        self._ax.vlines(up.index, up["Low"], up["High"], color="green", linestyle="-", lw=0.25)
        self._ax.vlines(down.index, down["Low"], down["High"], color="red", linestyle="-", lw=0.25)


class LinePlot(TickerPlot):
    def _plot(self, data: pd.DataFrame) -> None:
        self._ax.plot(data.index, data["Close"])


__all__ = ["LinePlot", "CandlestickPlot"]
