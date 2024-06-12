import torch
from torch.utils.data import Dataset

import pandas as pd


class TickerUpDownDataset(Dataset):
    def __init__(self, data: pd.DataFrame, lookahead: int = 7, label: str = "Close"):
        """
        Args:
            data:
            lookahead: intervals in the future used as the label
            label: column to use as label
        """
        self._lookahead = lookahead
        self._label = label
        self._data: pd.DataFrame = data.iloc[:-lookahead]

        y = data[label]
        y = y.shift(-lookahead) > y
        y = y[:-lookahead]

        x = self._data.copy()
        del x[label]

        self._y = torch.tensor(y.values, dtype=torch.float32)
        self._x = torch.tensor(x.values, dtype=torch.float32)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx: int | str) -> tuple[torch.Tensor, torch.Tensor] | pd.Series:
        if isinstance(idx, str):
            return self._data[idx]
        return self._x[idx], self._y[idx]

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y
