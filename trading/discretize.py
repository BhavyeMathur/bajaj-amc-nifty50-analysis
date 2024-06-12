from typing import Iterable

import pandas as pd
import numpy as np


def _as_numpy(data: Iterable):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pd.DataFrame):
        return data.values
    return np.array(data)


def discretize_bool(data: Iterable) -> np.ndarray:
    data = _as_numpy(data)
    return data.astype("bool")


def discretize_bin(data: Iterable, bins: int = 0, min_: None | float = None, max_: None | float = None) -> np.ndarray:
    data = _as_numpy(data)
    min_ = data.min() if min_ is None else min_
    max_ = data.max() if max_ is None else max_
    return np.digitize(data, bins=np.linspace(min_, max_, num=bins, endpoint=True))
