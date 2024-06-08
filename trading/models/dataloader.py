import torch
from torch.utils.data import Dataset
import pandas as pd

from trading import Ticker


class TickerUpDownDataset(Dataset):
    def __init__(self, ticker: Ticker, lookahead: int = 7, min_lookback: int = 0, max_lookback: int | None = None,
                 x: tuple[str, ...] = ("Close",), y: str = "Close",
                 period: str = "max", interval: str = "1d"):
        """
        Args:
            ticker:
            lookahead: intervals in the future used as the label
            min_lookback: minimum length of past data input x
            max_lookback: maximum length of past data input x
            x: columns to use as input
            y: column to use as label
            period:
            interval:
        """
        self._lookahead = lookahead
        self._min_lookback = min_lookback
        self._max_lookback = max_lookback

        self._label = y
        self._ticker = ticker
        self._data = ticker.history(period=period, interval=interval)

        self._set_y()
        self._data = self._data[list(x)]
        self._set_x()

    def __len__(self):
        return len(self._x) - self._min_lookback

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor] | pd.Series:
        if isinstance(idx, str):
            return self._data[idx]

        idx += self._min_lookback

        if self._max_lookback is None:
            return self._x[:idx], self._y[idx]
        return self._x[idx - self._max_lookback:idx], self._y[idx]

    def __setitem__(self, key: str, value):
        self._data[key] = value
        self._set_x()

    def __delitem__(self, key):
        del self._data[key]
        self._set_x()

    def _set_x(self):
        self._x = self._data.values
        self._x = self._x[:-self._lookahead]
        self._x = torch.tensor(self._x, dtype=torch.float32)

    def _set_y(self):
        self._y = self._data[self._label]
        self._y = self._y.shift(-self._lookahead) > self._y
        self._y = self._y[:-self._lookahead]
        self._y = torch.tensor(self._y.values, dtype=torch.float32)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._data.columns)

    @property
    def min_lookback(self) -> int:
        return self._min_lookback

    @property
    def max_lookback(self) -> int:
        return self._max_lookback

    @property
    def data(self) -> pd.DataFrame:
        return self._data

