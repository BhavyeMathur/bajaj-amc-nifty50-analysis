from trading import *
from torch.utils.data import DataLoader

NIFTY = YTicker("^NSEI")
data = models.TickerUpDownDataset(NIFTY)

train_dataloader = DataLoader(data, batch_size=1)
print(data[0])

