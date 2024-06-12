import pandas as pd
import numpy as np


class DiscreteState:
    def __init__(self, variable: str):
        self._variable = variable

    def __repr__(self) -> str:
        return f"DiscreteState('{self._variable}')"

    def setup(self, data: pd.Series | np.ndarray) -> None:
        return

    def discretize(self, data: pd.Series) -> pd.Series:
        """
        Returns a series translated into discrete states (ex. booleans or binned floats)
        """
        raise NotImplementedError()

    @property
    def variable(self) -> str:
        return self._variable


class Bool(DiscreteState):
    def __repr__(self) -> str:
        return f"Bool('{self._variable}')"

    def discretize(self, data):
        return data.astype("bool")


class Categorical(DiscreteState):
    def __repr__(self) -> str:
        return f"Categorical('{self._variable}')"

    def discretize(self, data):
        return data


class Bin(DiscreteState):
    def __init__(self, variable: str, bins: int = 10, min_=None, max_=None):
        super().__init__(variable)
        self._bins = bins
        self._min = min_
        self._max = max_

    def __repr__(self) -> str:
        return f"Bin('{self._variable}', bins={self._bins})"

    def setup(self, data) -> None:
        if self._min is None and self._max is None:
            raise RuntimeError("DiscreteState Bin already set up")

        if self._min is None:
            self._min = data.min()
        if self._max is None:
            self._max = data.max()

    def discretize(self, data):
        return np.digitize(data, bins=np.linspace(self._min, self._max, num=self._bins))
