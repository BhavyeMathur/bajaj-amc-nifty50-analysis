import torch
from torch.utils.data import Dataset
from trading import Ticker


class TickerUpDownDataset(Dataset):
    def __init__(self, ticker: Ticker, lookahead=7, y="Close", period="max", interval="1d", columns=("Close",)):
        self._ticker = ticker

        data = ticker.history(period=period, interval=interval)
        self._y = torch.tensor(data[y], dtype=torch.float32)
        self._x = torch.tensor(data[list(columns)].values, dtype=torch.float32)

        self._lookahead = lookahead

    def __len__(self):
        return len(self._x) - self._lookahead - 1

    def __getitem__(self, idx):
        idx += 1
        return (self._x[:idx],
                int(self._y[idx + self._lookahead] > self._y[idx]))
