from trading import *

from torch.utils.data import DataLoader
import torch.nn as nn

NIFTY = YTicker("^NSEI")
data = NIFTY.history(period=periods.max, interval=intervals.day)[["High", "Low", "Close", "Prev Close"]]

data["RSI"] = discretize_bin(RSIIndicator()(data))
data["SMA"] = discretize_bool(SMAIndicator(window=12)(data) > SMAIndicator(window=30)(data))
data["AO"] = discretize_bin(AwesomeOscillatorIndicator()(data))
data["Up"] = discretize_bool(data["Close"] > data["Prev Close"])
del data["High"], data["Low"], data["Prev Close"]
data.dropna(inplace=True)

dataset = models.TickerUpDownDataset(data, lookahead=1, label="Close")
dataloader = DataLoader(dataset, batch_size=len(dataset))

loss = nn.L1Loss()

model = models.MarkovClassifier(dataset)
tester = models.Backtest(model, dataloader, loss)
tester.test()

model = models.UpClassifier()
tester = models.Backtest(model, dataloader, loss)
tester.test()
