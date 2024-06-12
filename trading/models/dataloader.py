import torch
from torch.utils.data import Dataset
import pandas as pd

from trading import Ticker


class TickerUpDownDataset(Dataset):
    def __init__(self, ticker: Ticker, lookahead: int = 7, features: tuple[str, ...] = ("Close",), label: str = "Close",
                 period: str = "max", interval: str = "1d"):
        """
        Args:
            ticker:
            lookahead: intervals in the future used as the label
            features: columns to use as input
            label: column to use as label
            period:
            interval:
        """
        self._ticker = ticker
        self._lookahead = lookahead
        self._features = list(features)
        self._label = label

        self._data = ticker.history(period=period, interval=interval)

        y = self._data[self._label]
        y = y.shift(-lookahead) > y
        y = y[:-lookahead]
        self._y = torch.tensor(y.values, dtype=torch.float32)

        self._data = self._data[self._features].iloc[:-lookahead]
        self._data["y"] = y

        self._set_x()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx: int | str) -> tuple[torch.Tensor, torch.Tensor] | pd.Series:
        if isinstance(idx, str):
            return self._data[idx]
        return self._x[idx], self._y[idx]

    def __setitem__(self, key: str, value):
        self._features.append(key)
        self._data[key] = value
        self._set_x()

    def __delitem__(self, key):
        self._features.remove(key)
        del self._data[key]
        self._set_x()

    def _set_x(self):
        x = self._data[self._features].values
        self._x = torch.tensor(x.astype("float32"), dtype=torch.float32)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._data.columns)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def label(self) -> str:
        return self._label

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self._features)
