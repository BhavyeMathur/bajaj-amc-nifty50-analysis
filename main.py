from trading import *

from torch.utils.data import DataLoader
import torch.nn as nn

NIFTY = YTicker("^NSEI")
data = NIFTY.history(period=periods.halfyear, interval=intervals.day)[["High", "Low", "Close"]]

data["RSI"] = discretize_bin(RSIIndicator()(data))
del data["High"], data["Low"]
print(data.head(15))

data = models.TickerUpDownDataset(data, lookahead=1, label="Close")

# dataloader = DataLoader(data)
# loss = nn.L1Loss()
#
# model = models.MarkovClassifier(dataloader, discretize_bin("RSI"))
# tester = models.Backtest(model, dataloader, loss)
# tester.test()
#
# model = models.UpClassifier()
# tester = models.Backtest(model, dataloader, loss)
# tester.test()
