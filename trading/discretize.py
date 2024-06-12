from typing import Iterable

import pandas as pd
import numpy as np


def _as_numpy(data: Iterable):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pd.DataFrame):
        return data.values
    return np.array(data)


def _restore_nan(data, binned) -> np.ndarray:
    binned = binned.astype("float32")
    binned[np.isnan(data)] = np.nan
    return binned


def discretize_bool(data: Iterable) -> np.ndarray:
    binned = _as_numpy(data).astype("bool")
    return _restore_nan(data, binned)


def discretize_bin(data: Iterable, bins: int = 10, min_: None | float = None, max_: None | float = None) -> np.ndarray:
    data = _as_numpy(data)
    min_ = data.min() if min_ is None else min_
    max_ = data.max() if max_ is None else max_

    binned = np.digitize(data, bins=np.linspace(min_, max_, num=bins, endpoint=True))
    return _restore_nan(data, binned)


__all__ = ["discretize_bool", "discretize_bin"]
