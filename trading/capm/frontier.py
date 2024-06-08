from contextlib import contextmanager

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

from trading import Portfolio

pd.options.mode.chained_assignment = None


@contextmanager
def autoscale_turned_off(ax=None):
    ax = ax or plt.gca()
    lims = [ax.get_xlim(), ax.get_ylim()]
    yield
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])


class EfficientFrontier:
    def __init__(self, portfolio: Portfolio, frequency="1mo", period="max", risk_free_rate=0.07) -> None:
        import mplcyberpunk
        plt.style.use("cyberpunk")

        self._portfolio = portfolio
        self._frequency = frequency
        self._period = period
        self._risk_free_rate = risk_free_rate

        self._ax: plt.Axes | None = None

    def _sample_weights(self, n: int) -> np.ndarray:
        weights = np.random.uniform(0.0, 1.0, size=(n, len(self._portfolio)))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights

    def _sample_risk_reward(self, n: int) -> np.ndarray:
        weights = self._sample_weights(n)
        return self._portfolio.get_weighted_returns(weights, period=self._period, interval=self._frequency)

    def _setup_axes(self):
        fig = plt.figure()
        self._ax = fig.gca()

        self._ax.set_xlabel("Expected Risk", labelpad=15)
        self._ax.set_ylabel("Expected Reward", labelpad=15)
        self._ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))
        self._ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))
        self._ax.xaxis.set_tick_params(labelsize=6)
        self._ax.yaxis.set_tick_params(labelsize=6)

    def plot_position(self):
        y, x = self._portfolio.get_weighted_returns(self._portfolio.weights,
                                                    period=self._period, interval=self._frequency) * 100
        self._ax.scatter(x, y, c="#f5ac36")
        self.annotate(x + 0.02, y, "Current", fontsize=6)

    def annotate(self, x, y, label, **kwargs):
        kwargs = dict(fontsize=6, verticalalignment="center",
                      path_effects=[pe.withStroke(linewidth=2, foreground=self._ax.get_facecolor())]) | kwargs
        self._ax.annotate(label, (x, y), **kwargs)

    def plot(self, samples: int) -> None:
        y, x = self._sample_risk_reward(samples) * 100
        r = (self._risk_free_rate + 1) ** (1 / 12) - 1

        r *= 100

        self._setup_axes()

        # Risk-free return rate
        self._ax.scatter(0, r, c="#faac37")
        self.annotate(0.4, r, "T-Bills", fontsize=6)

        # Sampled portfolios
        self._ax.scatter(x, y, s=0.5, c="#71caf0")

        # Tangent portfolio
        sharpe = (y - r) / x
        maxx = x.max()

        # Individual stocks
        for stock in self._portfolio:
            y, x = stock.historical_returns(period=self._period, interval=self._frequency) * 100
            self._ax.scatter(x, y, c="#f51b4e")
            self.annotate(x, y, stock.symbol)

        with autoscale_turned_off(self._ax):
            self._ax.plot((0, maxx), (r, maxx * sharpe.max() + r), c="#faac37", linewidth=1, alpha=0.3)

    def compute_desired_returns(self, market):
        mkt_return = market.historical_returns(self._period, self._frequency)[0]
        r = (self._risk_free_rate + 1) ** (1 / 12) - 1

        data = {"stock": ["market"], "beta": [1], "desired": [100 * mkt_return], "actual": [100 * mkt_return]}

        for stock in self._portfolio:
            try:
                beta = stock.info["beta"]
            except KeyError:
                continue
            desired = 100 * (r + beta * (mkt_return - r))
            actual = 100 * stock.historical_returns(self._period, self._frequency)[0]

            data["stock"].append(stock.symbol)
            data["beta"].append(beta)
            data["desired"].append(desired)
            data["actual"].append(actual)

        print(pd.DataFrame(data))

    @staticmethod
    def show():
        plt.show()


__all__ = ["EfficientFrontier"]
