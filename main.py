from trading import *
import trading.models.markov as markov

from torch.utils.data import DataLoader
import torch.nn as nn

NIFTY = YTicker("^NSEI")

data = models.TickerUpDownDataset(NIFTY, period=periods.years5, lookahead=2, min_lookback=1, max_lookback=1,
                                  x=("Close", "Prev Close"))
data["Up"] = (data["Close"] > data["Prev Close"]).astype("float32")

dataloader = DataLoader(data)
loss = nn.L1Loss()

model = models.MarkovClassifier(dataloader, markov.Bool("Up"))
# model = models.UpClassifier()
tester = models.Backtest(model, dataloader, loss)
tester.test()
