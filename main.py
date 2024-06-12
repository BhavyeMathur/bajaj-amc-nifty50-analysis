from trading import *
import trading.models.markov as markov

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

NIFTY = YTicker("^NSEI")

data = models.TickerUpDownDataset(NIFTY, period=periods.years5, lookahead=1, min_lookback=1, max_lookback=1,
                                  x=("Close", "Prev Close"))

args = []
for t in np.arange(0.9, 1.1, 0.02):
    data[f"Up{t}"] = (data["Close"] > data["Prev Close"] * t).astype("float32")
    args.append(markov.Bool(f"Up{t}"))

dataloader = DataLoader(data, batch_size=1)
loss = nn.L1Loss()

model = models.MarkovClassifier(dataloader, *args)
tester = models.Backtest(model, dataloader, loss)
tester.test()
