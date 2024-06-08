from trading import *
from torch.utils.data import DataLoader
import torch.nn as nn

NIFTY = YTicker("^NSEI")

data = models.TickerUpDownDataset(NIFTY, period=periods.ytd, lookahead=2)
dataloader = DataLoader(data)

model = models.UpClassifier()
tester = models.Backtest(model, dataloader, nn.L1Loss())
tester.test()
