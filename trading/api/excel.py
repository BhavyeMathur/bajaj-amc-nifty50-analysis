import pandas as pd

from .ticker import Ticker


class ExcelTicker(Ticker):
    def __init__(self, ticker: str, filepath: str, sheet_name: str):
        super().__init__(ticker)
        self._filepath = filepath
        self._sheet_name = sheet_name

    def history(self) -> pd.DataFrame:
        d = pd.read_excel(self._filepath, sheet_name=self._sheet_name)
        d = d.set_index("Date", drop=True)
        d.index = pd.to_datetime(d.index)
        d = self._compute_derived_features(d)
        return d

    @property
    def currency(self) -> str:
        return "INR"
