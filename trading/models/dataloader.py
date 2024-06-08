import torch
from torch.utils.data import Dataset

from trading import Ticker


class TickerUpDownDataset(Dataset):
    def __init__(self, ticker: Ticker, lookahead: int = 7, min_lookback: int = 0, max_lookback: int | None = None,
                 x: tuple[str, ...] = ("Close",), y: str = "Close", period: str = "max", interval: str = "1d"):
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
        self._ticker = ticker
        data = ticker.history(period=period, interval=interval)

        self._y = data[y]
        self._y = self._y.shift(-lookahead) > self._y
        self._y = self._y[:-lookahead]
        self._y = torch.tensor(self._y, dtype=torch.float32)

        self._x = data[list(x)].values
        self._x = self._x[:-lookahead]
        self._x = torch.tensor(self._x, dtype=torch.float32)

        self._min_lookback = min_lookback
        self._max_lookback = max_lookback

    def __len__(self):
        return len(self._x) - self._min_lookback

    def __getitem__(self, idx):
        idx += self._min_lookback

        if self._max_lookback is None:
            return self._x[:idx], self._y[idx]
        return self._x[idx - self._max_lookback:idx], self._y[idx]
