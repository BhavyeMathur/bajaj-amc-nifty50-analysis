import sys

import pandas as pd
import numpy as np

from .stock import Stock


class Portfolio(list):
    def __init__(self, *args: Stock):
        super().__init__(args)
        self._stocks = {stock.name: stock for stock in args}
        self._weights = np.ones(len(self))
        self._returns = {}

    def __getitem__(self, item) -> Stock:
        if isinstance(item, str):
            return self._stocks[item]
        return super().__getitem__(item)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._stocks
        return super().__contains__(item)

    def __repr__(self) -> str:
        return f"Portfolio({super().__repr__()})"

    def pop(self, __index: int | str = -1) -> Stock:
        if isinstance(__index, str):
            stock = self._stocks.pop(__index)
            self.pop(self.index(stock))
            return stock
        return super().pop(__index)

    def index(self, __value, __start=0, __stop=sys.maxsize):
        if isinstance(__value, str):
            return super().index(self._stocks[__value])
        return super().index(__value, __start, __stop)

    def get_weighted_returns(self, weights, period="max", interval="1mo") -> np.ndarray:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        returns = self.history("% Change", period, interval)

        means = (returns + 1).mean() - 1
        expected = weights @ means

        weights = np.repeat(weights[..., None], len(self), axis=-1)
        weights *= np.swapaxes(weights, -2, -1)

        variance = weights * returns.cov().values
        variance = np.sum(variance, axis=(-2, -1))
        return np.array([expected, variance ** 0.5])

    def history(self, column: str, period="5y", interval="1mo"):
        if (key := (period, interval)) in self._returns:
            return self._returns[key]

        returns = pd.DataFrame({stock.name: stock.history(period, interval)[column] for stock in self})
        self._returns[key] = returns
        return returns

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, weights) -> None:
        self._weights = weights if isinstance(weights, np.ndarray) else np.array(weights)
        self._weights = self._weights.astype("float64")
        self._weights /= self._weights.sum()
